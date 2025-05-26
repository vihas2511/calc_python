#!/usr/bin/env python
import sys
import os
import datetime
import logging

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import *

# Additional library import for configuration if needed
from pg_composite_pipelines_configuration.configuration import Configuration

# Transfix functions and VFR expressions
from get_src_data import get_transfix as tvb
from load_vfr_us_ca import expr_vfr_us_ca as expr

# --- CONSTANTS (adjust these for your environment) ---
TARGET_DB_NAME = "target_db"         # Your target database name
RDS_DB_NAME = "rds_db"               # Your RDS source database name
TRANS_VB_DB_NAME = "trans_vb_db"     # Your Transfix source database name
SPARK_MASTER = "local[*]"            # Your Spark master setting
SPARK_CONF = [("spark.sql.shuffle.partitions", "8")]
DEBUG_SUFFIX = "_DEBUG"              # Suffix for debug table names

# For this consolidated script, extra parameters (staging_location, debug_mode_ind, debug_postfix) are not used.
# They are replaced with None.

# --- INLINE UTILITY FUNCTIONS ---

def manageOutput(log, spark, df, cache_ind, df_name, target_db):
    """
    Simplified version of manageOutput.
    If cache_ind == 1, registers the DataFrame as a temporary view and caches it.
    Otherwise, does nothing.
    """
    if cache_ind == 1:
        temp_view = df_name + DEBUG_SUFFIX.upper() + "_CACHE_VW"
        df.createOrReplaceTempView(temp_view)
        spark.sql(f"CACHE TABLE {temp_view}")
        log.info("Data frame '%s' cached as %s.", df_name, temp_view)
    return

def removeDebugTables(log, spark, target_db):
    """
    Remove any tables in target_db that end with DEBUG_SUFFIX (case-insensitive).
    """
    debug_postfix = DEBUG_SUFFIX.upper()
    spark.sql(f"USE {target_db}")
    tables = spark.sql("SHOW TABLES")
    log.info("Started dropping DEBUG tables.")
    for row in tables.collect():
        if row.tableName.upper().endswith(debug_postfix):
            spark.sql(f"DROP TABLE IF EXISTS {row.database}.{row.tableName}")
            log.info("Dropped table %s.%s", row.database, row.tableName)
    return

def get_spark_session(log, job_prefix, master, spark_config):
    """
    Create or retrieve a SparkSession using the provided configuration.
    """
    conf = pyspark.SparkConf()
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    app_name = f"{job_prefix}_{dt}_{os.getpid()}"
    conf.setAppName(app_name)
    conf.setMaster(master)
    log.info("Spark session configuration: %s", spark_config)
    for key, value in spark_config:
        conf.set(key, value)
    sparkSession = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
    log.info("Spark session established.")
    return sparkSession

def get_dbutils():
    """
    Dummy stub for dbutils – in environments like Databricks, this is provided.
    """
    return None

# --- INLINE GET_RDS FUNCTIONS (if needed for dimensions) ---

def get_cust_dim(log, spark, src_db_name):
    """
    Inlined function to get customer dimension data.
    """
    log.info("Started selecting cust_dim from %s.", src_db_name)
    sql_query = f"""
        SELECT cust_id,
               trade_chanl_curr_id AS trade_chanl_id
        FROM {src_db_name}.cust_dim
        WHERE curr_ind = 'Y'
    """
    df = spark.sql(sql_query)
    log.info("Selecting cust_dim from %s has finished.", src_db_name)
    return df

def get_trade_chanl_hier_dim(log, spark, src_db_name):
    """
    Inlined function to get trade channel hierarchy dimension data.
    """
    log.info("Started selecting trade_chanl_hier_dim from %s.", src_db_name)
    sql_query = f"""
        SELECT trade_chanl_2_long_name AS channel_name,
               trade_chanl_7_id AS trade_chanl_id
        FROM {src_db_name}.trade_chanl_hier_dim
        WHERE curr_ind = 'Y' AND trade_chanl_hier_id = '658'
    """
    df = spark.sql(sql_query)
    log.info("Selecting trade_chanl_hier_dim from %s has finished.", src_db_name)
    return df

# --- CORE FUNCTION: get_vfr_us_ca_star ---
def get_vfr_us_ca_star(log, spark):
    """
    Consolidated function that processes the VFR_US_CA data.
    It calls Transfix functions, applies expressions, aggregates data,
    and returns a final DataFrame ready for load.
    
    Extra parameters from the original version (staging_location, debug_mode_ind, debug_postfix)
    have been removed and replaced with constants.
    """
    # Retrieve the base VFR data using the Transfix function.
    vfr_data_df = tvb.get_vfr_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    )
    
    log.info("Calculate Bucket Values.")
    vfr_calc_df = vfr_data_df.withColumn(
            "oblb_gross_weight_plan_qty_load", f.expr(expr.oblb_gross_weight_plan_qty_expr)
        ).withColumn(
            "oblb_gross_weight_shipped_qty_load", f.expr(expr.oblb_gross_weight_shipped_qty_expr)
        ).withColumn(
            "oblb_net_weight_plan_qty_load", f.expr(expr.oblb_net_weight_plan_qty_expr)
        ).withColumn(
            "oblb_gross_vol_plan_qty_load", f.expr(expr.oblb_gross_vol_plan_qty_expr)
        ).withColumn(
            "oblb_gross_vol_shipped_qty_load", f.expr(expr.oblb_gross_vol_shipped_qty_expr)
        ).withColumn(
            "oblb_net_vol_plan_qty_load", f.expr(expr.oblb_net_vol_plan_qty_expr)
        ).withColumn(
            "fiscal_year_perd_num_int", f.col('fiscal_year_perd_num').cast("int")
        ).withColumn(
            "fiscal_year_num_int", f.col('fiscal_year_num').cast("int")
        ).filter("min_trans_stage_flag = 'Y'")
    
    manageOutput(log, spark, vfr_calc_df, 0, "vfr_calc_df", TARGET_DB_NAME)
    
    log.info("Aggregate gross values.")
    agg_gross_values = vfr_calc_df.select(
            "load_id", "total_gross_weight_qty", "oblb_gross_weight_plan_qty",
            "total_gross_vol_qty", "oblb_gross_vol_plan_qty", "pallet_spot_qty"
        ).distinct().groupBy("load_id").agg(
            f.sum("total_gross_weight_qty").alias("total_gross_weight_qty"),
            f.sum("oblb_gross_weight_plan_qty").alias("oblb_gross_weight_plan_qty"),
            f.sum("total_gross_vol_qty").alias("total_gross_vol_qty"),
            f.sum("oblb_gross_vol_plan_qty").alias("oblb_gross_vol_plan_qty"),
            f.sum("pallet_spot_qty").alias("pallet_spot_qty")
        )
    manageOutput(log, spark, agg_gross_values, 0, "agg_gross_values", TARGET_DB_NAME)
    
    log.info("Aggregate material level data.")
    agg_material_df = vfr_calc_df.select(
            "load_id", "load_gbu_id", "tdcval_code", "material_doc_num", "dlvry_item_num",
            "gross_weight_qty", "gross_vol_qty", "load_material_weight_qty", "load_material_vol_qty",
            "plan_net_weight_qty", "plan_net_vol_qty", "cut_impact_rate", "drf_last_truck_amt",
            "glb_segment_impact_cat_ld_amt", "hopade_amt", "max_orders_non_drp_amt", "max_orders_incrmtl_amt"
        ).distinct().join(agg_gross_values, "load_id", how='left').groupBy("load_id", "load_gbu_id").agg(
            f.sum("load_material_weight_qty").alias("load_material_weight_qty"),
            f.sum("load_material_vol_qty").alias("load_material_vol_qty"),
            f.sum("gross_weight_qty").alias("gross_weight_qty"),
            f.sum("gross_vol_qty").alias("gross_vol_qty"),
            f.max("total_gross_weight_qty").alias("total_gross_weight_qty"),
            f.max("oblb_gross_weight_plan_qty").alias("oblb_gross_weight_plan_qty"),
            f.max("total_gross_vol_qty").alias("total_gross_vol_qty"),
            f.max("oblb_gross_vol_plan_qty").alias("oblb_gross_vol_plan_qty"),
            f.max("pallet_spot_qty").alias("pallet_spot_qty"),
            f.sum("cut_impact_rate").alias("cut_impact_rate"),
            f.sum("drf_last_truck_amt").alias("drf_last_truck_amt"),
            f.sum("glb_segment_impact_cat_ld_amt").alias("glb_segment_impact_cat_ld_amt"),
            f.sum("hopade_amt").alias("hopade_amt"),
            f.sum("max_orders_non_drp_amt").alias("max_orders_non_drp_amt"),
            f.sum("max_orders_incrmtl_amt").alias("max_orders_incrmtl_amt"),
            f.max("plan_net_weight_qty").alias("plan_net_weight_qty"),
            f.max("plan_net_vol_qty").alias("plan_net_vol_qty")
        ).withColumn("agg_gross_weight_qty", f.expr(expr.agg_gross_weight_qty_expr)) \
         .withColumn("agg_gross_vol_qty", f.expr(expr.agg_gross_vol_qty_expr)) \
         .withColumn("agg_net_weight_qty", f.expr(expr.agg_net_weight_qty_expr)) \
         .withColumn("agg_net_vol_qty", f.expr(expr.agg_net_vol_qty_expr)) \
         .drop("plan_net_weight_qty").drop("plan_net_vol_qty")
    manageOutput(log, spark, agg_material_df, 0, "agg_material_df", TARGET_DB_NAME)
    log.info("Aggregate material level data has finished.")
    
    log.info("Calculating load level data.")
    agg_load_floor_position_df = vfr_calc_df.select("load_id", "floor_position_qty").distinct().groupBy("load_id").agg(
        f.sum("floor_position_qty").alias("floor_position_qty")
    )
    agg_load_pallet_df = vfr_calc_df.select("load_id", "pallet_qty").distinct().groupBy("load_id").agg(
        f.sum("pallet_qty").alias("pallet_qty")
    )
    agg_load_theortc_pallet_df = vfr_calc_df.select("load_id", "theortc_pallet_qty").distinct().groupBy("load_id").agg(
        f.sum("theortc_pallet_qty").alias("theortc_pallet_qty")
    )
    log.info("Calculating load level data has finished.")
    
    log.info("Calculating TFS subsector cost.")
    tfs_subsector_cost_df = tvb.get_tfs_subsector_cost_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    ).withColumn("load_id", f.regexp_replace(f.col('load_id'), '^0*', ''))\
     .withColumn("gbu_desc", f.expr(expr.gbu_desc_expr))\
     .withColumn("load_gbu_id", f.expr(expr.load_gbu_id_expr))\
     .withColumnRenamed("su_per_load_qty", "tfs_su_per_load_qty")\
     .withColumnRenamed("total_cost_amt", "tfs_total_cost_amt")\
     .withColumnRenamed("step_factor", "tfs_step_factor")\
     .drop("distance_per_load_num_qty").drop("gbu_desc") \
     .groupBy("load_id", "load_gbu_id").agg(
         f.sum("tfs_su_per_load_qty").alias("tfs_su_per_load_qty"),
         f.sum("tfs_total_cost_amt").alias("tfs_total_cost_amt"),
         f.sum("tfs_step_factor").alias("tfs_step_factor")
     )
    log.info("Calculating TFS subsector cost has finished.")
    
    log.info("Grouping VFR data.")
    final_tab_df = vfr_calc_df.drop("gross_weight_qty").drop("gross_vol_qty")\
        .drop("cut_impact_rate").drop("drf_last_truck_amt").drop("glb_segment_impact_cat_ld_amt")\
        .drop("hopade_amt").drop("max_orders_non_drp_amt").drop("max_orders_incrmtl_amt")\
        .groupBy("load_id", "gbu_desc", "gbu_code", "load_gbu_id").agg(
            f.max("weight_uom").alias("weight_uom"),
            f.max("weight_fill_above_100pct_flag").alias("weight_fill_above_100pct_flag"),
            f.max("vol_uom").alias("vol_uom"),
            f.max("vol_fill_above_100pct_flag").alias("vol_fill_above_100pct_flag"),
            f.max("vehicle_fill_rate_id").alias("vehicle_fill_rate_id"),
            f.max("vfr_last_update_utc_tmstp").alias("vfr_last_update_utc_tmstp"),
            f.max("tfts_load_tmstp").alias("tfts_load_tmstp"),
            f.max("stage_dest_point_id").alias("stage_dest_point_id"),
            f.max("stage_dprtr_point_code").alias("stage_dprtr_point_code"),
            f.max("sold_to_party_desc").alias("sold_to_party_desc"),
            f.max("sold_to_party_id").alias("sold_to_party_id"),
            f.max("ship_to_party_desc").alias("ship_to_party_desc"),
            f.max("ship_point_country_to_code").alias("ship_point_country_to_code"),
            f.max("ship_to_party_id").alias("ship_to_party_id"),
            f.max("ship_point_code").alias("ship_point_code"),
            f.max("ship_cond_desc").alias("ship_cond_desc"),
            f.max("ship_cond_val").alias("ship_cond_val"),
            f.max("recvng_live_drop_code").alias("recvng_live_drop_code"),
            f.sum("net_weight_qty").alias("net_weight_qty"),
            f.sum("net_vol_qty").alias("net_vol_qty"),
            f.max("max_weight_qty").alias("max_weight_qty"),
            f.max("max_vol_trans_mgmt_sys_qty").alias("max_vol_trans_mgmt_sys_qty"),
            f.max("max_pallet_tms_trans_type_qty").alias("max_pallet_tms_trans_type_qty"),
            f.max("ship_site_gbu_name").alias("ship_site_gbu_name"),
            f.max("vfr_freight_type_code").alias("vfr_freight_type_code"),
            f.max("floor_position_fill_rate_plan_qty").alias("floor_position_fill_rate_plan_qty"),
            f.max("flex_truck_order_desc").alias("flex_truck_order_desc"),
            f.max("fiscal_year_perd_num_int").alias("fiscal_year_perd_num"),
            f.max("fiscal_year_variant_code").alias("fiscal_year_variant_code"),
            f.max("fiscal_year_num_int").alias("fiscal_year_num"),
            f.max("load_from_file_name").alias("load_from_file_name"),
            f.max("external_id").alias("external_id"),
            f.max("pre_load_type_code").alias("pre_load_type_code"),
            f.max("distance_uom").alias("distance_uom"),
            f.max("distance_qty").alias("distance_qty"),
            f.max("country_from_code").alias("country_from_code"),
            f.max("default_ship_cond_code").alias("default_ship_cond_code"),
            f.max("customer_desc").alias("customer_desc"),
            f.max("customer_id").alias("customer_id"),
            f.max("country_to_name").alias("country_to_name"),
            f.max("country_to_code").alias("country_to_code"),
            f.max("country_from_desc").alias("country_from_desc"),
            f.max("carr_id").alias("carr_id"),
            f.max("carr_desc").alias("carr_desc"),
            f.max("shpmt_start_date").alias("shpmt_start_date"),
            f.max("actual_goods_issue_date").alias("actual_goods_issue_date"),
            f.max("gi_month_num").alias("gi_month_num"),
            f.max("tms_service_code").alias("tms_service_code"),
            f.max("weight_avg_qty").alias("weight_avg_qty"),
            f.max("density_rate").alias("density_rate"),
            f.max("shipped_load_cnt").alias("shipped_load_cnt"),
            f.max("total_load_cost_amt").alias("total_load_cost_amt"),
            f.sum("total_vf_optny_amt").alias("total_vf_optny_amt"),
            f.max("cases_impact_amt").alias("cases_impact_amt"),
            f.max("opertng_space_pct").alias("opertng_space_pct"),
            f.max("opertng_space_impact_amt").alias("opertng_space_impact_amt"),
            f.max("pallet_impact_amt").alias("pallet_impact_amt"),
            f.max("pallet_impact_pct").alias("pallet_impact_pct"),
            f.max("pallet_shipped_qty").alias("pallet_shipped_qty"),
            f.max("pallet_load_qty").alias("pallet_load_qty"),
            f.max("prod_density_gap_impact_amt").alias("prod_density_gap_impact_amt"),
            f.max("prod_density_gap_impact_pct").alias("prod_density_gap_impact_pct"),
            f.max("max_net_weight_order_qty").alias("max_net_weight_order_qty"),
            f.max("max_net_vol_qty").alias("max_net_vol_qty"),
            f.max("net_density_order_qty").alias("net_density_order_qty"),
            f.max("net_vol_fill_rate").alias("net_vol_fill_rate"),
            f.max("net_vol_order_qty").alias("net_vol_order_qty"),
            f.max("net_weight_order_qty").alias("net_weight_order_qty"),
            f.max("follow_on_doc_num").alias("follow_on_doc_num"),
            f.max("pallet_num_qty").alias("pallet_num_qty"),
            f.max("release_date").alias("release_date"),
            f.max("release_datetm").alias("release_datetm"),
            f.max("truck_type_code").alias("truck_type_code"),
            f.max("vehicle_trans_medium_code").alias("vehicle_trans_medium_code"),
            f.max("vehicle_axle_position_front_val").alias("vehicle_axle_position_front_val"),
            f.max("vehicle_axle_position_back_val").alias("vehicle_axle_position_back_val"),
            f.max("vehicle_max_axle_weight_front_qty").alias("vehicle_max_axle_weight_front_qty"),
            f.max("vehicle_max_axle_weight_back_qty").alias("vehicle_max_axle_weight_back_qty"),
            f.max("vehicle_inner_length_val").alias("vehicle_inner_length_val"),
            f.max("vehicle_inner_width_val").alias("vehicle_inner_width_val"),
            f.max("vehicle_inner_height_val").alias("vehicle_inner_height_val"),
            f.max("vehicle_floorspot_footprint_num_val").alias("vehicle_floorspot_footprint_num_val"),
            f.max("vehicle_floorspot_width_val").alias("vehicle_floorspot_width_val"),
            f.max("vehicle_floorspot_length_val").alias("vehicle_floorspot_length_val"),
            f.max("vehicle_name").alias("vehicle_name"),
            f.max("vehicle_min_back_axle_position_qty").alias("vehicle_min_back_axle_position_qty"),
            f.max("doc_flow_load_id").alias("doc_flow_load_id"),
            f.max("ordered_shipped_flag").alias("ordered_shipped_flag"),
            f.max("low_density_site_val").alias("low_density_site_val"),
            f.max("customer_lvl2_code").alias("customer_lvl2_code"),
            f.max("customer_lvl2_desc").alias("customer_lvl2_desc"),
            f.max("customer_lvl3_code").alias("customer_lvl3_code"),
            f.max("customer_lvl3_desc").alias("customer_lvl3_desc"),
            f.max("customer_lvl4_code").alias("customer_lvl4_code"),
            f.max("customer_lvl4_desc").alias("customer_lvl4_desc"),
            f.max("customer_lvl5_code").alias("customer_lvl5_code"),
            f.max("customer_lvl5_desc").alias("customer_lvl5_desc"),
            f.max("customer_lvl6_code").alias("customer_lvl6_code"),
            f.max("customer_lvl6_desc").alias("customer_lvl6_desc"),
            f.max("customer_lvl7_code").alias("customer_lvl7_code"),
            f.max("customer_lvl7_desc").alias("customer_lvl7_desc"),
            f.max("customer_lvl8_code").alias("customer_lvl8_code"),
            f.max("customer_lvl8_desc").alias("customer_lvl8_desc"),
            f.max("customer_lvl9_code").alias("customer_lvl9_code"),
            f.max("customer_lvl9_desc").alias("customer_lvl9_desc"),
            f.max("customer_lvl10_code").alias("customer_lvl10_code"),
            f.max("customer_lvl10_desc").alias("customer_lvl10_desc"),
            f.max("customer_lvl11_code").alias("customer_lvl11_code"),
            f.max("customer_lvl11_desc").alias("customer_lvl11_desc"),
            f.max("customer_lvl12_code").alias("customer_lvl12_code"),
            f.max("customer_lvl12_desc").alias("customer_lvl12_desc"),
            f.max("actual_carr_total_trans_cost_usd_amt").alias("actual_carr_total_trans_cost_usd_amt"),
            f.max("linehaul_cost_amt").alias("linehaul_cost_amt"),
            f.max("incrmtl_freight_auction_cost_amt").alias("incrmtl_freight_auction_cost_amt"),
            f.max("cnc_carr_mix_cost_amt").alias("cnc_carr_mix_cost_amt"),
            f.max("unsource_cost_amt").alias("unsource_cost_amt"),
            f.max("fuel_cost_amt").alias("fuel_cost_amt"),
            f.max("acsrl_cost_amt").alias("acsrl_cost_amt"),
            f.max("sambc_flag").alias("sambc_flag"),
            f.max("origin_zone_ship_from_code").alias("origin_zone_ship_from_code"),
            f.max("dest_ship_from_code").alias("dest_ship_from_code"),
            f.max("dest_loc_code").alias("dest_loc_code"),
            f.max("freight_auction_flag").alias("freight_auction_flag"),
            f.max("tac_freight_type_code").alias("tac_freight_type_code"),
            f.max("origin_freight_code").alias("origin_freight_code"),
            f.max("step_factor").alias("step_factor"),
            f.max("trans_dest_exectn_shpmt_end_date").alias("trans_dest_exectn_shpmt_end_date"),
            f.max("trans_dest_plan_shpmt_end_date").alias("trans_dest_plan_shpmt_end_date"),
            f.max("trans_dest_request_dlvry_date").alias("trans_dest_request_dlvry_date"),
            f.max("trans_origin_exectn_checkin_date").alias("trans_origin_exectn_checkin_date"),
            f.max("trans_origin_plan_checkin_date").alias("trans_origin_plan_checkin_date"),
            f.max("trans_plan_shpmt_start_date").alias("trans_plan_shpmt_start_date"),
            f.max("last_update_utc_tmstp").alias("last_update_utc_tmstp"),
            f.max("load_density_rate").alias("load_density_rate"),
            f.max("su_per_load_cnt").alias("su_per_load_cnt"),
            f.max("shipped_net_vol_qty").alias("shipped_net_vol_qty"),
            f.max("shipped_net_weight_qty").alias("shipped_net_weight_qty"),
            f.max("shipped_gross_vol_qty").alias("shipped_gross_vol_qty"),
            f.max("shipped_gross_weight_qty").alias("shipped_gross_weight_qty"),
            f.max("combined_load_max_weight_qty").alias("combined_load_max_weight_qty"),
            f.max("combined_load_max_vol_qty").alias("combined_load_max_vol_qty"),
            f.max("plan_gross_weight_qty").alias("plan_gross_weight_qty"),
            f.max("plan_gross_vol_qty").alias("plan_gross_vol_qty"),
            f.max("plan_net_weight_qty").alias("plan_net_weight_qty"),
            f.max("plan_net_vol_qty").alias("plan_net_vol_qty"),
            f.max("load_builder_prty_val").alias("load_builder_prty_val")
        )\
        .join(agg_load_floor_position_df, "load_id")\
        .join(agg_load_pallet_df, "load_id")\
        .join(agg_load_theortc_pallet_df, "load_id")\
        .join(agg_material_df, ["load_id", "load_gbu_id"])\
        .join(tfs_subsector_cost_df, ["load_id", "load_gbu_id"], "left_outer")
    
    manageOutput(log, spark, final_tab_df, 0, "final_tab_df", target_db_name)
    log.info("Group by VFR data has finished.")
    
    return final_tab_df


def load_vfr_us_ca_star(log):
    """
    Consolidated load function for VFR_US_CA.
    Creates a Spark session, removes debug tables,
    reads the target schema, processes the VFR_US_CA data via get_vfr_us_ca_star,
    and writes the DataFrame to the target table "vfr_load_agg_star" using overwrite.
    """
    spark = get_spark_session(log, "tfx_vfr_us_ca", SPARK_MASTER, SPARK_CONF)
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    removeDebugTables(log, spark, TARGET_DB_NAME)
    
    log.info("Started loading {}.vfr_load_agg_star table.".format(TARGET_DB_NAME))
    
    # Get target table column list – assumes the target table already exists.
    target_table_cols = spark.table(f"{TARGET_DB_NAME}.vfr_load_agg_star").schema.fieldNames()
    
    # Get the processed VFR_US_CA data.
    vfr_us_ca_star_df = get_vfr_us_ca_star(log, spark).select(target_table_cols)
    
    log.info("Inserting data into {}.vfr_load_agg_star (overwriting old data)".format(TARGET_DB_NAME))
    vfr_us_ca_star_df.write.insertInto(f"{TARGET_DB_NAME}.vfr_load_agg_star", overwrite=True)
    log.info("Loading {}.vfr_load_agg_star table has finished".format(TARGET_DB_NAME))
    return

# --- MAIN FUNCTION ---
def main():
    dbutils = get_dbutils()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Optionally load external configuration (ignored here in favor of constants)
    config = Configuration.load_for_default_environment(__file__, dbutils)
    
    logger.info("Starting load for VFR_US_CA star table.")
    load_vfr_us_ca_star(logger)

if __name__ == "__main__":
    main()
