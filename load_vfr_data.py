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

# Import Transfix functions and VFR expressions
from get_src_data import get_transfix as tvb
from load_vfr import expr_vfr as expr

# --- CONSTANTS (Update these for your environment) ---
TARGET_DB_NAME = "target_db"       # Your target database name
RDS_DB_NAME = "rds_db"             # Your RDS source database name
TRANS_VB_DB_NAME = "trans_vb_db"   # Your Transfix source database name
SPARK_MASTER = "local[*]"          # Your Spark master setting
SPARK_CONF = [("spark.sql.shuffle.partitions", "8")]
DEBUG_SUFFIX = "_DEBUG"            # Suffix for debug table names (if needed)

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
    Remove any tables in target_db that end with DEBUG_SUFFIX (case insensitive).
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
    Dummy stub for dbutils – in environments like Databricks, this is provided automatically.
    """
    return None

# --- INLINE GET_RDS FUNCTIONS ---

def get_cust_dim(log, spark, src_db_name):
    """
    Retrieves customer dimension data (inlined from rds.get_cust_dim).
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
    Retrieves trade channel hierarchy dimension data (inlined from rds.get_trade_chanl_hier_dim).
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

# --- LOAD FUNCTION: VFR Data Hub Star ---
def load_vfr_data_hub_star(log):
    """
    Creates a Spark session, removes debug tables,
    retrieves the target table schema, processes VFR data,
    and writes the resulting DataFrame into the target table.
    """
    spark = get_spark_session(log, "tfx_vfr", SPARK_MASTER, SPARK_CONF)
    removeDebugTables(log, spark, TARGET_DB_NAME)
    
    log.info("Started loading {}.vfr_data_hub_star table.".format(TARGET_DB_NAME))
    
    # Get customer and trade channel dimensions from RDS
    cust_dim_df = get_cust_dim(log, spark, RDS_DB_NAME)
    trade_chanl_hier_dim_df = get_trade_chanl_hier_dim(log, spark, RDS_DB_NAME)
    
    log.info("Creating trade channel hierarchy data.")
    trade_chanl_df = cust_dim_df.join(trade_chanl_hier_dim_df, "trade_chanl_id", "inner")\
        .withColumnRenamed("cust_id", "customer_lvl12_code")
    manageOutput(log, spark, trade_chanl_df, 1, "trade_chanl_df", TARGET_DB_NAME)
    
    # Get VFR data hub star
    vfr_data_hub_star_df = tvb.get_vfr_data_hub_star(log, spark, TRANS_VB_DB_NAME, TARGET_DB_NAME)
    
    # Get target table column list (assumes that the target table already exists)
    target_table_cols = spark.table(f"{TARGET_DB_NAME}.vfr_data_hub_star").schema.fieldNames()
    
    vfr_final_df = vfr_data_hub_star_df.select(target_table_cols)
    
    log.info("Inserting data into {}.vfr_data_hub_star (overwriting old data)".format(TARGET_DB_NAME))
    vfr_final_df.write.insertInto(f"{TARGET_DB_NAME}.vfr_data_hub_star", overwrite=True)
    log.info("Loading {}.vfr_data_hub_star table has finished.".format(TARGET_DB_NAME))
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
    
    config = Configuration.load_for_default_environment(__file__, dbutils)
    
    logger.info("Starting load for VFR data hub star table.")
    load_vfr_data_hub_star(logger)

if __name__ == "__main__":
    main()
