#!/usr/bin/env python
import sys
import os
import datetime
import logging

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import *

# Additional library import for configuration (if needed)
from pg_composite_pipelines_configuration.configuration import Configuration

# Transfix functions and network scorecard expressions
from get_src_data import get_transfix as tvb
from load_netw_scorec import expr_network_scorecard as expr

# --- CONSTANTS (adjust these for your environment) ---
TARGET_DB_NAME = "target_db"       # Replace with your target database name
TRANS_VB_DB_NAME = "trans_vb_db"   # Replace with your Transfix (source) database name
SPARK_MASTER = "local[*]"          # Adjust for your environment
SPARK_CONF = [("spark.sql.shuffle.partitions", "8")]
DEBUG_SUFFIX = "_DEBUG"            # Suffix for debug table names

# For this consolidated version, extra parameters (staging_location, debug_mode_ind, debug_postfix)
# are replaced by None.
STAGING_LOCATION = None
DEBUG_MODE = None
DEBUG_POSTFIX = None

# --- INLINE UTILITY FUNCTIONS ---

def manageOutput(log, spark, df, cache_ind, df_name, target_db):
    """
    A simplified version of manageOutput.
    If cache_ind == 1, the DataFrame is registered as a temporary view and cached.
    Otherwise, no caching is done.
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
    Create or get a SparkSession using the provided configuration.
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
    Dummy stub for dbutils. In environments like Databricks, dbutils is provided automatically.
    """
    return None

# --- CORE FUNCTION: get_weekly_network_sccrd_star ---
def get_weekly_network_sccrd_star(log, spark, trans_vsblt_db, target_db, staging_loc, debug_mode, debug_postfix):
    """
    Consolidated function to process the weekly network scorecard data.
    Extra parameters are removed (staging_loc, debug_mode, debug_postfix are set to None).
    """
    # Retrieve the TAC tender pg summary data from Transfix.
    tac_tender_pg_summary_df = tvb.get_tac_tender_pg_summary_new(
        log, spark, 'ap_transfix_tv_na', target_db, staging_loc, debug_mode, debug_postfix
    )\
    .withColumn("carr_num", f.expr(expr.tendigit_carrier_id_expr))\
    .withColumnRenamed("carr_num", "long_forward_agent_id")\
    .withColumnRenamed("calendar_year_week_tac", "week_year_val")\
    .withColumnRenamed("carrier_id", "forward_agent_id")\
    .withColumn("lane_key", f.concat(f.col("origin_zone_ship_from_code"), f.lit("-"), f.col("dest_ship_from_code")))\
    .withColumnRenamed("tms_service_code", "service_tms_code")\
    .withColumnRenamed("origin_location_id", "origin_code")\
    .drop("carr_desc")
    # (Optional filtering commented out)
    
    # Retrieve monster view data (first version)
    monster_data_df = tvb.get_monster_view_data(
        log, spark, target_db, target_db, staging_loc, debug_mode, debug_postfix
    )\
    .withColumn("lane_name", f.concat(f.col("parent_loc_code"), f.lit("-"), f.col("dest_ship_from_code")))\
    .withColumnRenamed("ship_week_num", "week_year_val")\
    .groupBy("parent_loc_code", "dest_ship_from_code", "ship_to_party_code", "carr_num", "service_tms_code",
             "origin_code", "origin_zone_ship_from_code", "week_year_val")\
    .agg(
        f.max("ship_to_party_desc").alias("ship_to_party_desc"),
        f.max("dest_sold_to_name").alias("dest_sold_to_name"),
        f.max("carr_desc").alias("carr_desc"),
        f.max("actual_ship_date").alias("actual_ship_date"),
        f.max("freight_type_val").alias("freight_type_val"),
        f.max("cdot_ontime_cnt").alias("cdot_ontime_cnt"),
        f.sum("shpmt_cnt").alias("shpmt_cnt"),
        f.sum("shpmt_on_time_cnt").alias("shpmt_on_time_cnt"),
        f.sum("measrbl_shpmt_cnt").alias("measrbl_shpmt_cnt"),
        f.max("parent_carr_name").alias("parent_carr_name"),
        f.max("lane_name").alias("lane_name")
    )\
    .withColumnRenamed("carr_num", "forward_agent_id")\
    .drop("origin_zone_ship_from_code")\
    .drop("ship_to_party_code")
    
    # Retrieve monster_data 2 (second version)
    monster_data_2_df = tvb.get_monster_view_data(
        log, spark, target_db, target_db, staging_loc, debug_mode, debug_postfix
    )\
    .withColumn("lane_name", f.concat(f.col("parent_loc_code"), f.lit("-"), f.col("dest_ship_from_code")))\
    .withColumnRenamed("ship_week_num", "week_year_val")\
    .groupBy("dest_ship_from_code", "carr_num", "service_tms_code", "origin_code", "week_year_val")\
    .agg(
        f.max("ship_to_party_desc").alias("ship_to_party_desc"),
        f.max("dest_sold_to_name").alias("dest_sold_to_name"),
        f.max("carr_desc").alias("carr_desc"),
        f.max("actual_ship_date").alias("actual_ship_date"),
        f.max("freight_type_val").alias("freight_type_val"),
        f.max("cdot_ontime_cnt").alias("cdot_ontime_cnt"),
        f.sum("shpmt_cnt").alias("shpmt_cnt"),
        f.sum("shpmt_on_time_cnt").alias("shpmt_on_time_cnt"),
        f.sum("measrbl_shpmt_cnt").alias("measrbl_shpmt_cnt"),
        f.max("parent_carr_name").alias("parent_carr_name"),
        f.max("ship_to_party_code").alias("ship_to_party_code"),
        f.max("origin_zone_ship_from_code").alias("origin_zone_ship_from_code"),
        f.max("parent_loc_code").alias("parent_loc_code"),
        f.max("lane_name").alias("lane_name")
    )\
    .withColumnRenamed("carr_num", "forward_agent_id")\
    .drop("origin_zone_ship_from_code")\
    .drop("ship_to_party_code")\
    .withColumn("monster_origin_code", f.col("origin_code"))\
    .withColumnRenamed("week_year_val", "week_year_val_mon")\
    .withColumnRenamed("forward_agent_id", "forward_agent_id_mon")\
    .withColumnRenamed("service_tms_code", "service_tms_code_mon")\
    .withColumnRenamed("dest_ship_from_code", "dest_ship_from_code_mon")\
    .withColumnRenamed("origin_code", "origin_code_mon")
    
    # Retrieve lot view data
    lot_data_df = tvb.get_lot_view_data(
        log, spark, target_db, target_db, staging_loc, debug_mode, debug_postfix
    )\
    .withColumn("lane_name", f.concat(f.col("origin_zone_code"), f.lit("-"), f.col("dest_ship_from_code")))\
    .withColumnRenamed("ship_week_num", "week_year_val")
    
    # Retrieve aggregated VFR data
    vfr_data_df = tvb.get_vfr_agg_data(
        log, spark, 'ap_transfix_tv_na', target_db, staging_loc, debug_mode, debug_postfix
    )\
    .withColumn("lane_name", f.concat(f.col("origin_zone_ship_from_code"), f.lit("-"), f.col("dest_loc_code"))) \
    .withColumn("carrier_id", f.regexp_replace(f.col('carr_id'), '^0*', ''))\
    .withColumn("formatted_actl_ship_date", f.concat(
            f.substring(f.col("shpmt_start_date"), 7, 4),
            f.lit("-"),
            f.substring(f.col("shpmt_start_date"), 4, 2),
            f.lit("-"),
            f.substring(f.col("shpmt_start_date"), 1, 2)
        ))\
    .withColumn("actl_ship_weekofyear", f.weekofyear(f.col("formatted_actl_ship_date")))\
    .withColumn("actl_ship_week", f.concat(f.col("actl_ship_weekofyear"), f.lit("/"),
                                           f.substring(f.col("shpmt_start_date"), 7, 4)))\
    .withColumn("calendar_week_num", f.lpad(f.weekofyear("formatted_actl_ship_date"), 2, '0'))\
    .withColumn("calendar_year_num", f.year("formatted_actl_ship_date"))\
    .withColumn("week_year_val", f.concat(f.col("calendar_week_num"), f.lit("/"), f.col("calendar_year_num")))
    
    manageOutput(log, spark, vfr_data_df, 0, "vfr_data_df", target_db)
    
    # --- Join Processing ---
    log.info("Joining data (2) - origin id.")
    netw_scorec2_1_df = monster_data_df.withColumn("monster_origin_code", f.col("origin_code"))\
        .join(tac_tender_pg_summary_df,
              ["week_year_val", "forward_agent_id", "service_tms_code", "origin_code", "dest_ship_from_code"],
              'left_outer')
    log.info("Joining data (2) - origin id has finished.")
    manageOutput(log, spark, netw_scorec2_1_df, 0, "netw_scorec2_1_df_new", target_db,
                 staging_loc, debug_mode, "_{}{}".format(target_db, debug_postfix))
    
    netw_scorec2_2_df = tac_tender_pg_summary_df\
        .join(monster_data_2_df,
              (monster_data_2_df.week_year_val_mon == tac_tender_pg_summary_df.week_year_val) &
              (monster_data_2_df.forward_agent_id_mon == tac_tender_pg_summary_df.forward_agent_id) &
              (monster_data_2_df.service_tms_code_mon == tac_tender_pg_summary_df.service_tms_code) &
              (monster_data_2_df.dest_ship_from_code_mon == tac_tender_pg_summary_df.dest_ship_from_code) &
              (monster_data_2_df.origin_code_mon == tac_tender_pg_summary_df.origin_code),
              how='right')\
         .filter((tac_tender_pg_summary_df.week_year_val.isNull()) &
                 (tac_tender_pg_summary_df.forward_agent_id.isNull()) & 
                 (tac_tender_pg_summary_df.service_tms_code.isNull()) &
                 (tac_tender_pg_summary_df.origin_code.isNull()) & 
                 (tac_tender_pg_summary_df.dest_ship_from_code.isNull()))
    log.info("Joining data (2) - origin id has finished.")
    manageOutput(log, spark, netw_scorec2_2_df, 0, "netw_scorec2_2_df", target_db,
                 staging_loc, debug_mode, "_{}{}".format(target_db, debug_postfix))
    
    netw_scorec_rename_df = netw_scorec2_2_df\
        .drop("week_year_val").drop("forward_agent_id")\
        .drop("service_tms_code").drop("dest_ship_from_code")\
        .drop("origin_code")\
        .withColumnRenamed("week_year_val_mon", "week_year_val")\
        .withColumnRenamed("forward_agent_id_mon", "forward_agent_id")\
        .withColumnRenamed("service_tms_code_mon", "service_tms_code")\
        .withColumnRenamed("dest_ship_from_code_mon", "dest_ship_from_code")\
        .withColumnRenamed("origin_code_mon", "origin_code")
    
    log.info("Joining data (2) - lane.")
    netw_scorec2_3_df = netw_scorec_rename_df\
        .groupBy("parent_loc_code", "dest_ship_from_code", "ship_to_party_code", "forward_agent_id", "service_tms_code", "week_year_val")\
        .agg(f.max("origin_zone_ship_from_code").alias("origin_zone_ship_from_code"),
             f.max("ship_to_party_desc").alias("ship_to_party_desc"), 
             f.max("dest_sold_to_name").alias("dest_sold_to_name"),
             f.max("actual_ship_date").alias("actual_ship_date"), 
             f.max("freight_type_val").alias("freight_type_val"), 
             f.max("cdot_ontime_cnt").alias("cdot_ontime_cnt"),
             f.max("shpmt_cnt").alias("shpmt_cnt"),
             f.max("shpmt_on_time_cnt").alias("shpmt_on_time_cnt"),
             f.max("measrbl_shpmt_cnt").alias("measrbl_shpmt_cnt"),
             f.max("parent_carr_name").alias("parent_carr_name"),
             f.max("lane_name").alias("lane_name")
             )\
        .drop("origin_zone_ship_from_code")\
        .withColumnRenamed("parent_loc_code", "origin_zone_ship_from_code")
    
    log.info("Joining data (2) - origin id has finished.")
    manageOutput(log, spark, netw_scorec2_3_df, 0, "netw_scorec2_3_df", target_db,
                 staging_loc, debug_mode, "_{}{}".format(target_db, debug_postfix))
    
    netw_scorec2_4_df = tac_tender_pg_summary_df\
        .join(netw_scorec_rename_df.withColumn("parent_loc_code", f.col("origin_zone_ship_from_code")),
              ["week_year_val", "forward_agent_id", "service_tms_code", "origin_zone_ship_from_code", "dest_ship_from_code"],
              'right_outer')
    log.info("Joining data (2) - origin id has finished.")
    manageOutput(log, spark, netw_scorec2_4_df, 0, "netw_scorec2_4_df", target_db,
                 staging_loc, debug_mode, "_{}{}".format(target_db, debug_postfix))
    
    netw_scorec2_df = netw_scorec2_1_df.select("week_year_val", "lane_key", "campus_lane_name", "forward_agent_id", "service_tms_code",
                                 "origin_code", "monster_origin_code", "actual_carr_trans_cost_amt", "linehaul_cost_amt",
                                 "incrmtl_freight_auction_cost_amt", "cnc_carr_mix_cost_amt",
                                 "unsource_cost_amt", "fuel_cost_amt", "acsrl_cost_amt", "lane",
                                 "origin_zone_ship_from_code", "dest_ship_from_code", "carr_desc",
                                 "sold_to_party_id", "avg_award_weekly_vol_qty", "ship_cond_val",
                                 "country_from_code", "country_to_code", "freight_auction_flag",
                                 "primary_carr_flag", "week_begin_date", "month_type_val", "cal_year_num",
                                 "month_date", "week_num", "region_code", "accept_cnt", "total_cnt",
                                 "dest_postal_code", "reject_cnt", "accept_pct", "reject_pct",
                                 "expct_vol_val", "reject_below_award_val", "weekly_carr_rate",
                                 "customer_desc", "customer_code", "customer_lvl3_desc",
                                 "customer_lvl5_desc", "customer_lvl6_desc", "customer_lvl12_desc",
                                 "customer_specific_lane_name", "long_forward_agent_id",
                                 "parent_loc_code", "ship_to_party_desc", "dest_sold_to_name",
                                 "actual_ship_date", "freight_type_val", "cdot_ontime_cnt", "shpmt_cnt",
                                 "shpmt_on_time_cnt", "measrbl_shpmt_cnt", "parent_carr_name") \
        .union(
        netw_scorec2_4_df.select("week_year_val", "lane_key", "campus_lane_name", "forward_agent_id", "service_tms_code",
                                 "origin_code", "monster_origin_code", "actual_carr_trans_cost_amt", "linehaul_cost_amt",
                                 "incrmtl_freight_auction_cost_amt", "cnc_carr_mix_cost_amt", "unsource_cost_amt",
                                 "fuel_cost_amt", "acsrl_cost_amt",
                                 "lane", "origin_zone_ship_from_code", "dest_ship_from_code", "carr_desc",
                                 "sold_to_party_id", "avg_award_weekly_vol_qty", "ship_cond_val", "country_from_code",
                                 "country_to_code", "freight_auction_flag", "primary_carr_flag", "week_begin_date",
                                 "month_type_val", "cal_year_num", "month_date", "week_num", "region_code",
                                 "accept_cnt", "total_cnt", "dest_postal_code", "reject_cnt", "accept_pct",
                                 "reject_pct", "expct_vol_val", "reject_below_award_val", "weekly_carr_rate",
                                 "customer_desc", "customer_code", "customer_lvl3_desc", "customer_lvl5_desc",
                                 "customer_lvl6_desc", "customer_lvl12_desc", "customer_specific_lane_name",
                                 "long_forward_agent_id", "parent_loc_code", "ship_to_party_desc", "dest_sold_to_name",
                                 "actual_ship_date", "freight_type_val", "cdot_ontime_cnt", "shpmt_cnt",
                                 "shpmt_on_time_cnt", "measrbl_shpmt_cnt", "parent_carr_name"))
    
    log.info("Joining data (2) - final has finished.")
    manageOutput(log, spark, netw_scorec2_df, 0, "netw_scorec2_df", target_db,
                 staging_loc, debug_mode, "_{}{}".format(target_db, debug_postfix))
    
    netw_scorec_filter_df = netw_scorec2_df.where(netw_scorec2_df.monster_origin_code != 'NULL')
    
    netw_scorec3_df = lot_data_df.drop("load_id").drop("actual_ship_date").drop("freight_type_val")\
        .groupBy("carr_num", "dest_ship_from_code", "ship_point_code", 
                 "actual_service_tms_code", "week_year_val") \
        .agg(f.max("carr_desc").alias("carr_desc_lot"),
             f.max("dest_zone_val").alias("dest_zone_val_lot"),
             f.max("origin_zone_code").alias("origin_zone_code_lot"),
             f.max("origin_zone_ship_from_code").alias("origin_zone_ship_from_code_lot"),
             f.max("ship_to_party_code").alias("ship_to_party_code_lot"),
             f.max("ship_to_party_desc").alias("ship_to_party_desc_lot"),
             f.sum("lot_otd_cnt").alias("lot_otd_count"),
             f.sum("lot_tat_late_counter_val").alias("lot_tat_late_counter_val"),
             f.sum("lot_cust_failure_cnt").alias("lot_customer_failure_cnt"),
             f.sum("pg_failure_cnt").alias("pg_failure_cnt"),
             f.sum("carr_failure_cnt").alias("carr_failure_cnt"),
             f.sum("others_failure_cnt").alias("others_failure_cnt"),
             f.max("tolrnc_sot_val").alias("tolrnc_sot_val")
             ) \
        .withColumnRenamed("ship_point_code", "monster_origin_code") \
        .withColumnRenamed("actual_service_tms_code", "service_tms_code") \
        .withColumnRenamed("carr_num", "forward_agent_id") \
        .drop("carr_desc_lot").drop("dest_zone_val_lot") \
        .drop("origin_zone_code_lot").drop("origin_zone_ship_from_code_lot") \
        .drop("ship_to_party_code_lot").drop("ship_to_party_desc_lot")
    
    log.info("Joining data (3) has finished.")
    manageOutput(log, spark, netw_scorec3_df, 0, "netw_scorec3_df", target_db,
                 staging_loc, debug_mode, "_{}{}".format(target_db, debug_postfix))
    
    log.info("Joining data (4).")
    netw_scorec4_df = netw_scorec_filter_df\
        .join(netw_scorec3_df, ["monster_origin_code", "dest_ship_from_code", "forward_agent_id", "service_tms_code", "week_year_val"], 'left_outer')
    log.info("Joining data (4) has finished.")
    manageOutput(log, spark, netw_scorec4_df, 0, "netw_scorec4_df", target_db,
                 staging_loc, debug_mode, "_{}{}".format(target_db, debug_postfix))
    
    log.info("Joining data (5).")
    netw_scorec5_df = vfr_data_df.drop("load_id").drop("shpmt_start_date").drop("freight_type_val")\
        .groupBy("ship_point_code", "dest_ship_from_code", "carrier_id", "tms_service_code", "actl_ship_week")\
        .agg(f.max("carr_desc").alias("carr_desc_vfr"),
             f.max("origin_loc_id").alias("origin_loc_id_vfr"),
             f.max("lane_name").alias("lane_name"),
             f.sum("su_per_load_cnt").alias("su_per_load_cnt"),
             f.sum("plan_gross_weight_qty").alias("plan_gross_weight_qty"),
             f.sum("plan_net_weight_qty").alias("plan_net_weight_qty"),
             f.sum("shipped_gross_weight_qty").alias("shipped_gross_weight_qty"),
             f.sum("shipped_net_weight_qty").alias("shipped_net_weight_qty"),
             f.sum("plan_gross_vol_qty").alias("plan_gross_vol_qty"),
             f.sum("plan_net_vol_qty").alias("plan_net_vol_qty"),
             f.sum("shipped_gross_vol_qty").alias("shipped_gross_vol_qty"),
             f.sum("shipped_net_vol_qty").alias("shipped_net_vol_qty"),
             f.sum("max_weight_qty").alias("max_weight_qty"),
             f.sum("max_vol_trans_mgmt_sys_qty").alias("max_vol_trans_mgmt_sys_qty"),
             f.avg("plan_gross_weight_qty").alias("avg_plan_gross_weight_qty"),
             f.avg("plan_net_weight_qty").alias("avg_plan_net_weight_qty"),
             f.avg("shipped_gross_weight_qty").alias("avg_shipped_gross_weight_qty"),
             f.avg("shipped_net_weight_qty").alias("avg_shipped_net_weight_qty"),
             f.avg("plan_gross_vol_qty").alias("avg_plan_gross_vol_qty"),
             f.avg("plan_net_vol_qty").alias("avg_plan_net_vol_qty"),
             f.avg("shipped_gross_vol_qty").alias("avg_shipped_gross_vol_qty"),
             f.avg("shipped_net_vol_qty").alias("avg_shipped_net_vol_qty"),
             f.avg("max_weight_qty").alias("avg_max_weight_qty"),
             f.avg("max_vol_trans_mgmt_sys_qty").alias("avg_max_vol_trans_mgmt_sys_qty"),
             f.avg("floor_position_qty").alias("floor_position_qty"),
             f.avg("max_pallet_tms_trans_type_qty").alias("max_pallet_tms_trans_type_qty"),
             f.sum("cut_impact_rate").alias("cut_impact_rate"),
             f.sum("drf_last_truck_amt").alias("drf_last_truck_amt"),
             f.sum("glb_segment_impact_cat_ld_amt").alias("glb_segment_impact_cat_ld_amt"),
             f.max("prod_density_gap_impact_amt").alias("prod_density_gap_impact_amt"),
             f.max("prod_density_gap_impact_pct").alias("prod_density_gap_impact_pct"),
             f.max("max_net_weight_order_qty").alias("max_net_weight_order_qty"),
             f.max("max_net_vol_qty").alias("max_net_vol_qty"),
             f.max("net_density_order_qty").alias("net_density_order_qty"),
             f.max("net_vol_fill_rate").alias("net_vol_fill_rate"),
             f.max("net_vol_order_qty").alias("net_vol_order_qty"),
             f.max("net_weight_order_qty").alias("net_weight_order_qty")
             )\
        .withColumnRenamed("lane_name", "lane_key") \
        .withColumnRenamed("tms_service_code", "service_tms_code") \
        .withColumnRenamed("carrier_id", "forward_agent_id") \
        .withColumnRenamed("ship_point_code", "monster_origin_code") \
        .withColumnRenamed("actl_ship_week", "week_year_val")\
        .drop("dest_loc_code")\
        .drop("carr_desc_vfr").drop("origin_loc_id_vfr")\
        .drop("dest_ship_from_code_vfr")\
        .drop("origin_zone_ship_from_code")
    log.info("Joining data (5) has finished.")
    manageOutput(logging, spark, netw_scorec5_df, 0, "netw_scorec5_df", target_db_name,
                 staging_loc, debug_mode, "_{}{}".format(target_db_name, debug_postfix))
    
    log.info("Joining data (6).")
    netw_scorec6_df = netw_scorec4_df\
        .join(netw_scorec5_df,
              ["monster_origin_code", "dest_ship_from_code", "forward_agent_id", "service_tms_code", "week_year_val"],
              'left_outer')\
        .withColumnRenamed("lane", "lane_name") \
        .drop("lane_key").drop("origin_loc_id_vfr")
    log.info("Joining data (6) has finished.")
    manageOutput(logging, spark, netw_scorec6_df, 0, "netw_scorec6_df", target_db_name,
                 staging_loc, debug_mode, "_{}{}".format(target_db_name, debug_postfix))
    
    return netw_scorec6_df

def load_network_scorecard(log, config_module, debug_mode_ind, debug_postfix):
    """
    Consolidated load function for the weekly_network_sccrd_agg_star table.
    It creates a Spark session, removes debug tables, obtains the target schema,
    processes the network scorecard data via get_weekly_network_sccrd_star, and writes
    the resulting DataFrame into the target table (overwriting old data).
    """
    # For this inline version, we remove extra parameters in favor of constants.
    spark = get_spark_session(log, 'tfx_wns', SPARK_MASTER, SPARK_CONF)
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    removeDebugTables(log, spark, TARGET_DB_NAME)
    
    log.info("Started loading {}.weekly_network_sccrd_agg_star table.".format(TARGET_DB_NAME))
    
    # Get target table columns (assumes the target table already exists)
    target_table_cols = spark.table(f"{TARGET_DB_NAME}.weekly_network_sccrd_agg_star").schema.fieldNames()
    
    weekly_network_sccrd_star_df = get_weekly_network_sccrd_star(
        log, spark, TRANS_VB_DB_NAME, TARGET_DB_NAME, None, None, None
    ).select(target_table_cols)
    
    log.info("Inserting data into {}.weekly_network_sccrd_agg_star (overwriting old data)".format(TARGET_DB_NAME))
    weekly_network_sccrd_star_df.write.insertInto(f"{TARGET_DB_NAME}.weekly_network_sccrd_agg_star", overwrite=True)
    log.info("Loading {}.weekly_network_sccrd_agg_star table has finished".format(TARGET_DB_NAME))
    return

def main():
    dbutils = get_dbutils()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Optionally load external configuration (ignored here in favor of constants)
    config = None  # or you could use: Configuration.load_for_default_environment(__file__, dbutils)
    
    logger.info("Starting load for weekly network scorecard agg star table.")
    load_network_scorecard(logger, config, None, None)

if __name__ == "__main__":
    main()
