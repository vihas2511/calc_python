#!/usr/bin/env python
import sys
import os
import datetime
import logging

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window
# Additional type functions if needed
from pyspark.sql.types import *
from pyspark.sql.functions import col

# Import the Transfix functions and LOT expressions
from get_src_data import get_transfix as tvb
from load_lot import expr_lot as expr

# Additional configuration library (if needed)
from pg_composite_pipelines_configuration.configuration import Configuration

# --- CONSTANTS (adjust these as needed) ---
TARGET_DB_NAME = "target_db"     # Replace with your target database name
SPARK_MASTER = "local[*]"         # Adjust for your environment
SPARK_CONF = [("spark.sql.shuffle.partitions", "8")]
DEBUG_SUFFIX = "_DEBUG"           # Suffix used for debug table names

# --- INLINE UTILITY FUNCTIONS ---
def manageOutput(log, spark, df, cache_ind, df_name, target_db):
    """
    A simplified version of manageOutput.
    If cache_ind == 1, registers the DataFrame as a temporary view and caches it.
    For cache_ind == 0, does nothing.
    """
    if cache_ind == 1:
        temp_view = df_name + DEBUG_SUFFIX.upper() + "_CACHE_VW"
        df.createOrReplaceTempView(temp_view)
        spark.sql(f"CACHE TABLE {temp_view}")
        log.info("Data frame '%s' cached as %s.", df_name, temp_view)
    return

def removeDebugTables(log, spark, target_db):
    """
    Remove any tables in the target database that end with DEBUG_SUFFIX (case insensitive).
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
    Create or get a SparkSession with the given configuration.
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
    Dummy stub for dbutils. In environments such as Databricks, dbutils is provided.
    """
    return None

# --- LOT STAR FUNCTIONS ---
def get_lot_star(log, spark):
    """
    Consolidated function that loads and processes the LOT data.
    Extra parameters have been removed in favor of constant values.
    """
    log.info("Get child shipment.")
    # Get base data from on_time_data_hub_star using tvb; note that we now pass constants.
    lot_child_shpmt_df = tvb.get_on_time_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    ).select("child_shpmt_num") \
      .withColumn("load_child_id", f.regexp_replace("child_shpmt_num", '^0', '')) \
      .drop("child_shpmt_num")
    log.info("Get child shipment completed.")
    
    lot_otd_dh_df = tvb.get_on_time_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    )\
    .drop("pg_order_num").drop("cust_po_num").drop("request_dlvry_from_date")\
    .drop("request_dlvry_from_datetm").drop("request_dlvry_to_date").drop("request_dlvry_to_datetm")\
    .drop("orig_request_dlvry_from_tmstp").drop("orig_request_dlvry_to_tmstp").drop("actual_arrival_datetm")\
    .drop("load_method_num").drop("transit_mode_name").drop("order_create_date").drop("order_create_datetm")\
    .drop("schedule_date").drop("schedule_datetm").drop("tender_date").drop("tender_datetm")\
    .drop("first_dlvry_appt_date").drop("first_dlvry_appt_datetm").drop("last_dlvry_appt_date")\
    .drop("last_dlvry_appt_datetm").drop("actual_ship_datetm").drop("csot_failure_reason_bucket_name")\
    .drop("csot_failure_reason_bucket_updated_name").drop("measrbl_flag").drop("profl_method_code")\
    .drop("dest_city_name").drop("dest_state_code").drop("dest_postal_code").drop("cause_code")\
    .drop("on_time_src_code").drop("on_time_src_desc").drop("sales_org_code")\
    .drop("sd_doc_item_overall_process_status_val").drop("multi_stop_num")\
    .drop("lot_exception_categ_val").drop("primary_carr_flag").drop("actual_shpmt_end_date")\
    .drop("trnsp_stage_num").drop("actual_dlvry_tmstp").drop("first_dlvry_appt_tmstp")\
    .filter('trnsp_stage_num = "1"')\
    .filter('LENGTH(COALESCE(actual_load_end_date, "")) > 1')\
    .filter('LENGTH(COALESCE(final_lrdt_date, "")) > 1')\
    .filter('LENGTH(COALESCE(lot_delay_reason_code_desc, "")) > 1')\
    .filter('change_type_code in ("PICKCI", "PICKCO")') \
    .filter('LENGTH(COALESCE(event_datetm, "")) > 1') \
    .filter('event_datetm <> "99991231235959"')
    
    lot_otd_dh_filter_df = lot_otd_dh_df.join(
        lot_child_shpmt_df,
        lot_otd_dh_df.load_id == lot_child_shpmt_df.load_child_id,
        how='left'
    ).filter('load_child_id is null')
    
    log.info("Calculating new columns for on time data hub.")
    lot_cols_otd_dh_df = lot_otd_dh_filter_df\
        .withColumnRenamed("dest_zone_code", "dest_zone_val")\
        .withColumnRenamed("frt_type_desc", "freight_type_val") \
        .withColumn("ship_month", f.expr(expr.ship_month_expr))\
        .withColumn("ship_year", f.expr(expr.ship_year_expr)) \
        .withColumn("ship_3lettermonth", f.expr(expr.ship_3letter_expr)) \
        .withColumn("lot_otd_cnt", f.expr(expr.lot_otd_count_expr)) \
        .withColumn("lot_tat_late_counter_val", f.expr(expr.lot_tat_late_counter_expr)) \
        .withColumn("country_to_desc", f.expr(expr.cntry_to_desc_expr)) \
        .withColumn("actual_ship_month_val", f.concat(f.col('ship_3lettermonth'), f.lit(' '), f.col('ship_year')))
    
    manageOutput(log, spark, lot_cols_otd_dh_df, 1, "lot_cols_otd_dh_df", TARGET_DB_NAME)
    log.info("Calculating new columns for on time data hub has finished.")
    
    log.info("Group by load_id to calculate no of loads.")
    lot_cnt_otd_dh_df = lot_cols_otd_dh_df.select("load_id").groupBy("load_id")\
        .agg(f.countDistinct("load_id").alias("load_num_per_load_id_cnt"))
    log.info("Group by load_id to calculate no of loads has finished.")
    
    log.info("Get Max values for load_id.")
    lot_final_otd_dh_df = lot_cols_otd_dh_df.join(lot_cnt_otd_dh_df, "load_id", how='left')\
        .withColumn("rn",
            f.row_number().over(
                Window.partitionBy("load_id", "lot_delay_reason_code", "change_type_code")
                      .orderBy(f.col("event_datetm").desc())
            )
        ).filter("rn = 1")\
        .withColumn("rn2",
            f.row_number().over(
                Window.partitionBy("load_id", "change_type_code")
                      .orderBy(f.col("event_datetm").desc())
            )
        ).filter("rn2 = 1")\
        .withColumn("last_update_utc_tmstp", f.to_utc_timestamp(f.from_unixtime(f.unix_timestamp()), 'PRT'))
    
    manageOutput(log, spark, lot_final_otd_dh_df, 0, "lot_final_otd_dh_df", TARGET_DB_NAME)
    log.info("Get Max values for load_id has finished.")
    
    return lot_final_otd_dh_df

def load_lot_star(log):
    """
    Loads the lot_star table by creating a Spark session, removing
    debug tables, retrieving the target table schema, processing the LOT
    data via get_lot_star, and writing the resulting DataFrame (overwriting
    old data) into the target table.
    """
    spark = get_spark_session(log, 'tfx_lot', SPARK_MASTER, SPARK_CONF)
    removeDebugTables(log, spark, TARGET_DB_NAME)
    
    log.info("Started loading {}.lot_star table.".format(TARGET_DB_NAME))
    
    # Get target table column list (assumes that the target table already exists)
    target_table_cols = spark.table(f"{TARGET_DB_NAME}.lot_star").schema.fieldNames()
    
    lot_star_df = get_lot_star(log, spark).select(target_table_cols)
    
    log.info("Inserting data into {}.lot_star (overwriting old data)".format(TARGET_DB_NAME))
    lot_star_df.write.insertInto(f"{TARGET_DB_NAME}.lot_star", overwrite=True)
    log.info("Loading {}.lot_star table has finished.".format(TARGET_DB_NAME))
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
    
    # Optionally load configuration if needed (here we ignore external configuration)
    config = Configuration.load_for_default_environment(__file__, dbutils)
    
    logger.info("Starting load for LOT star table.")
    load_lot_star(logger)

if __name__ == "__main__":
    main()
