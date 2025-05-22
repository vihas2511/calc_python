#!/usr/bin/env python
import sys
import os
import datetime
import logging

import pyspark
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

# Additional libraries per request
from pg_composite_pipelines_configuration.configuration import Configuration

# Transfix functions and expressions remain as before
from get_src_data import get_transfix as tvb

# --- CONSTANTS (update these for your environment) ---
TARGET_DB_NAME = "target_db"       # Replace with your target database name
TRANS_VB_DB_NAME = "trans_vb_db"   # Replace with your Transfix database name
SPARK_MASTER = "local[*]"          # Adjust for your environment
SPARK_CONF = [("spark.sql.shuffle.partitions", "8")]  # Example spark config

# --- INLINE UTILITY FUNCTIONS (from your utility file) ---

def manageOutput(log, spark, df, cache_ind, df_name, target_db_name):
    """
    Simplified version of manageOutput.
    When cache_ind == 1, registers a temporary view and caches the DataFrame.
    For cache_ind == 0, no caching is performed.
    """
    if cache_ind == 1:
        temp_view = df_name + "_DEBUG_CACHE_VW"
        df.createOrReplaceTempView(temp_view)
        spark.sql(f"CACHE TABLE {temp_view}")
        log.info("Data frame '%s' cached as %s.", df_name, temp_view)
    return

def removeDebugTables(log, spark, target_db_name):
    """
    Remove any tables in target_db_name that end with '_DEBUG' (case insensitive).
    """
    debug_postfix = "_DEBUG"
    spark.sql(f"USE {target_db_name}")
    tables = spark.sql("SHOW TABLES")
    log.info("Started dropping DEBUG tables.")
    for row in tables.collect():
        if row.tableName.upper().endswith(debug_postfix.upper()):
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
    Stub for dbutils.
    In an environment like Databricks, dbutils is preloaded. Here, we just return None.
    """
    return None

# --- INLINE CARRIER FUNCTIONS (consolidated and simplified) ---

def get_tac_lane_detail_star(log, spark, trans_vsblt_db_name, target_db_name):
    """
    Loads and processes the TAC lane detail star data.
    Extra parameters have been removed.
    """
    # Call the Transfix function to get the summary data.
    tac_tender_pg_summary_df = tvb.get_tac_tender_pg_summary_new_star(
        log, spark, target_db_name, target_db_name, None, None, None
    ).drop("actual_carr_trans_cost_amt")\
     .drop("linehaul_cost_amt")\
     .drop("incrmtl_freight_auction_cost_amt")\
     .drop("cnc_carr_mix_cost_amt")\
     .drop("unsource_cost_amt")\
     .drop("fuel_cost_amt")\
     .drop("acsrl_cost_amt")\
     .drop("forward_agent_id")\
     .drop("service_tms_code")\
     .drop("sold_to_party_id")\
     .drop("ship_cond_val")\
     .drop("primary_carr_flag")\
     .drop("month_type_val")\
     .drop("cal_year_num")\
     .drop("month_date")\
     .drop("week_num")\
     .drop("dest_postal_code")\
     .withColumnRenamed("region_code", "state_province_code")\
     .distinct()

    log.info("Calculating columns in TAC tender pg summary data.")
    tac_tender_pg_summary_calc_df = tac_tender_pg_summary_df\
        .withColumn("calendar_week_num", f.weekofyear("week_begin_date"))\
        .withColumn("calendar_year_num", f.year("week_begin_date"))\
        .withColumn("str_calendar_week_num", f.lpad("calendar_week_num", 2, '0'))\
        .withColumn("concat_week_year", f.concat(f.col('calendar_year_num'), f.col('str_calendar_week_num')))\
        .withColumn("drank_week_year", f.dense_rank().over(Window.orderBy(f.col("concat_week_year").desc())))\
        .withColumn("last_update_utc_tmstp", f.to_utc_timestamp(f.from_unixtime(f.unix_timestamp()), 'PRT'))
    
    manageOutput(log, spark, tac_tender_pg_summary_calc_df, 0, "tac_tender_pg_summary_calc_df", target_db_name)
    log.info("Calculating columns in TAC tender pg summary data has finished.")
    
    return tac_tender_pg_summary_calc_df.filter('drank_week_year < 14')

def get_tac_shpmt_detail_star(log, spark, trans_vsblt_db_name, target_db_name):
    """
    Loads and processes TAC shipment detail data.
    Extra parameters have been removed.
    """
    tac_df = tvb.get_tac(log, spark, target_db_name, target_db_name, None, None, None)\
        .withColumn("load_id", f.regexp_replace("load_id", '^0', ''))\
        .drop("actual_carr_trans_cost_amt")\
        .drop("linehaul_cost_amt")\
        .drop("incrmtl_freight_auction_cost_amt")\
        .drop("cnc_carr_mix_cost_amt")\
        .drop("unsource_cost_amt")\
        .drop("fuel_cost_amt")\
        .drop("acsrl_cost_amt")\
        .drop("applnc_subsector_step_cnt")\
        .drop("baby_care_subsector_step_cnt")\
        .drop("chemical_subsector_step_cnt")\
        .drop("fabric_subsector_step_cnt")\
        .drop("family_subsector_step_cnt")\
        .drop("fem_subsector_step_cnt")\
        .drop("hair_subsector_step_cnt")\
        .drop("home_subsector_step_cnt")\
        .drop("oral_subsector_step_cnt")\
        .drop("phc_subsector_step_cnt")\
        .drop("shave_subsector_step_cnt")\
        .drop("skin_subsector_cnt")\
        .drop("other_subsector_cnt")\
        .drop("customer_lvl1_code")\
        .drop("customer_lvl1_desc")\
        .drop("customer_lvl2_code")\
        .drop("customer_lvl2_desc")\
        .drop("customer_lvl3_code")\
        .drop("customer_lvl3_desc")\
        .drop("customer_lvl4_code")\
        .drop("customer_lvl4_desc")\
        .drop("customer_lvl5_code")\
        .drop("customer_lvl5_desc")\
        .drop("customer_lvl6_code")\
        .drop("customer_lvl6_desc")\
        .drop("customer_lvl7_code")\
        .drop("customer_lvl7_desc")\
        .drop("customer_lvl8_code")\
        .drop("customer_lvl8_desc")\
        .drop("customer_lvl9_code")\
        .drop("customer_lvl9_desc")\
        .drop("customer_lvl10_code")\
        .drop("customer_lvl10_desc")\
        .drop("customer_lvl11_code")\
        .drop("customer_lvl11_desc")\
        .drop("customer_lvl12_code")\
        .drop("customer_lvl12_desc")\
        .drop("origin_zone_code")\
        .drop("daily_award_qty")\
        .distinct()
    
    tac_lane_detail_star_df = tvb.get_tac_lane_detail_star(
        log, spark, target_db_name, target_db_name, None, None, None
    ).distinct()
    
    shipping_location_na_dim_df = tvb.get_shipping_location_na_dim(
        log, spark, trans_vsblt_db_name, target_db_name, None, None, None
    ).drop("origin_zone_ship_from_code")\
     .drop("loc_id")\
     .drop("loc_name")\
     .withColumnRenamed("postal_code", "final_stop_postal_code")\
     .distinct()

    log.info("Calculating columns in TAC data.")
    cd_shpmt_tac_calc_df = tac_df\
        .withColumn("calendar_week_num", f.weekofyear("actual_goods_issue_date"))\
        .withColumn("calendar_year_num", f.year("actual_goods_issue_date"))\
        .withColumn("str_calendar_week_num", f.lpad("calendar_week_num", 2, '0'))\
        .withColumn("concat_week_year", f.concat(f.col('calendar_year_num'), f.col('str_calendar_week_num')))\
        .drop("calendar_week_num").drop("calendar_year_num").drop("str_calendar_week_num")
    log.info("Calculating columns in TAC data has finished.")
    
    log.info("Calculating columns in TAC Lane Detail data.")
    cd_shpmt_tac_lane_detail_star_calc_df = tac_lane_detail_star_df\
        .withColumn("calendar_week_num", f.weekofyear("week_begin_date"))\
        .withColumn("calendar_year_num", f.year("week_begin_date"))\
        .withColumn("str_calendar_week_num", f.lpad("calendar_week_num", 2, '0'))\
        .withColumn("concat_week_year", f.concat(f.col('calendar_year_num'), f.col('str_calendar_week_num')))\
        .drop("calendar_week_num").drop("calendar_year_num").drop("str_calendar_week_num")
    log.info("Calculating columns in TAC Lane Detail data has finished.")
    
    log.info("Joining tables for final data.")
    cd_shpmt_join_df = cd_shpmt_tac_calc_df\
        .join(cd_shpmt_tac_lane_detail_star_calc_df, "concat_week_year", "inner")\
        .join(shipping_location_na_dim_df, "final_stop_postal_code", "left")\
        .withColumn("last_update_utc_tmstp", f.to_utc_timestamp(f.from_unixtime(f.unix_timestamp()), 'PRT'))
    manageOutput(log, spark, cd_shpmt_join_df, 0, "cd_shpmt_join_df", target_db_name)
    log.info("Joining tables for final data has finished.")
    
    return cd_shpmt_join_df

def load_carrier_dashboard(log):
    """
    Creates a Spark session, removes any debug tables,
    processes carrier dashboard tables, and writes the results
    to the target tables by overwriting old data.
    """
    # Create a Spark session using our inline get_spark_session
    spark = get_spark_session(log, "tfx_carrier", SPARK_MASTER, SPARK_CONF)
    
    # Remove debug tables from the target DB
    removeDebugTables(log, spark, TARGET_DB_NAME)
    
    # --- Process TAC Lane Detail Star ---
    log.info("Started loading %s.tac_lane_detail_star table.", TARGET_DB_NAME)
    # Get target table schema to select only valid columns
    target_cols = spark.table(f"{TARGET_DB_NAME}.tac_lane_detail_star").schema.fieldNames()
    tac_lane_detail_star_df = get_tac_lane_detail_star(log, spark, TRANS_VB_DB_NAME, TARGET_DB_NAME)\
                                .select(*target_cols)
    log.info("Inserting data into %s.tac_lane_detail_star (overwriting old data).", TARGET_DB_NAME)
    tac_lane_detail_star_df.write.insertInto(f"{TARGET_DB_NAME}.tac_lane_detail_star", overwrite=True)
    log.info("Loading %s.tac_lane_detail_star table has finished.", TARGET_DB_NAME)
    
    # --- Process TAC Shipment Detail Star ---
    log.info("Started loading %s.tac_shpmt_detail_star table.", TARGET_DB_NAME)
    target_cols = spark.table(f"{TARGET_DB_NAME}.tac_shpmt_detail_star").schema.fieldNames()
    tac_shpmt_detail_star_df = get_tac_shpmt_detail_star(log, spark, TRANS_VB_DB_NAME, TARGET_DB_NAME)\
                                .select(*target_cols)
    log.info("Inserting data into %s.tac_shpmt_detail_star (overwriting old data).", TARGET_DB_NAME)
    tac_shpmt_detail_star_df.write.insertInto(f"{TARGET_DB_NAME}.tac_shpmt_detail_star", overwrite=True)
    log.info("Loading %s.tac_shpmt_detail_star table has finished.", TARGET_DB_NAME)
    
    return

# --- MAIN FUNCTION ---

def main():
    # Set up a dummy dbutils (or use the platform-provided one)
    dbutils = get_dbutils()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # (Optionally load a configuration. Here we could load one using the Configuration class)
    config = Configuration.load_for_default_environment(__file__, dbutils)
    
    # For this sample we ignore the configuration module and directly use constants.
    logger.info("Starting load for carrier dashboard tables.")
    
    load_carrier_dashboard(logger)

if __name__ == "__main__":
    main()
