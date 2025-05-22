#!/usr/bin/env python
import sys
import os
import datetime
import logging

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import LongType, DecimalType
from pyspark.sql.functions import col

# Additional library import per request
from pg_composite_pipelines_configuration.configuration import Configuration

# Transfix functions and CSOT expressions
from get_src_data import get_transfix as tvb
from load_csot import expr_csot as expr

# --- CONSTANTS (adjust these for your environment) ---
TARGET_DB_NAME = "target_db"       # Replace with your target database name
TRANS_VB_DB_NAME = "trans_vb_db"   # Replace with your Transfix (source) database name
SPARK_MASTER = "local[*]"          # Adjust or replace with your Spark master setting
SPARK_CONF = [("spark.sql.shuffle.partitions", "8")]
DEBUG_SUFFIX = "_DEBUG"            # Suffix used for debug table names

# --- INLINE UTILITY FUNCTIONS ---

def manageOutput(log, spark, df, cache_ind, df_name, target_db):
    """
    Simplified manageOutput that, if cache_ind==1, registers the DataFrame as a temporary view and caches it.
    If cache_ind==0, does nothing.
    """
    if cache_ind == 1:
        temp_view = df_name + DEBUG_SUFFIX.upper() + "_CACHE_VW"
        df.createOrReplaceTempView(temp_view)
        spark.sql(f"CACHE TABLE {temp_view}")
        log.info("Data frame '%s' cached as %s.", df_name, temp_view)
    return

def removeDebugTables(log, spark, target_db):
    """
    Remove any tables in the target database that end with DEBUG_SUFFIX (case-insensitive).
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
    Dummy stub for dbutils. In environments like Databricks, dbutils is provided.
    """
    return None

# --- CSOT FUNCTIONS ---

def get_csot_star(log, spark):
    """
    Consolidated function that loads and processes the CSOT data.
    All extra parameters have been removed; constant values are used.
    """
    # Get CSOT base data from on_time_data_hub_star.
    csot_data_df = tvb.get_on_time_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    ).drop("actual_dlvry_tmstp").drop("first_appt_dlvry_tmstp")\
     .drop("plan_shpmt_start_date").drop("plant_code").drop("transit_mode_name")\
     .drop("sold_to_party_code").drop("trans_plan_point_code").drop("plan_shpmt_start_datetm")\
     .drop("cpu_for_hire_desc").drop("change_type_code").drop("drop_live_ind_code").drop("child_shpmt_num")\
     .drop("user_tid_val").drop("lot_exception_categ_desc").drop("change_type_desc").drop("freight_auction_flag")\
     .drop("parent_shpmt_flag").drop("drop_live_ind_desc")
    
    csot_data_group1_df = tvb.get_on_time_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    ).select("load_id", "trnsp_stage_num", "event_datetm").distinct()
    
    csot_data_group2_df = tvb.get_on_time_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    ).select("load_id", "trnsp_stage_num", "event_datetm", "actual_dlvry_tmstp", "first_appt_dlvry_tmstp").distinct()
    
    csot_data_drop_live_ind_df = tvb.get_on_time_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    ).select("load_id", "drop_live_ind_desc").distinct()
    
    origin_gbu_df = tvb.get_origin_gbu_lkp(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    )
    
    dest_channel_cust_df = tvb.get_destination_channel_customer_lkp(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    )
    
    ship_to_pgp_flag_df = tvb.get_ship_to_pgp_flag_lkp(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    )
    
    on_time_codes_aot_reason_df = tvb.get_on_time_codes_aot_reason_lkp(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    )
    
    tfs_df = tvb.get_tfs(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None
    ).select("shpmt_id", "distance_per_load_qty")\
     .withColumn("loadid", f.regexp_replace(f.col('shpmt_id'), '^0*', ''))\
     .drop("shpmt_id").withColumnRenamed("loadid", "load_id")
    
    log.info("Calculating destination sold to and lot delay codes.")
    step1_df = csot_data_df.withColumn("dest_sold_to_name", f.expr(expr.dest_sold_to_name_expr)) \
        .withColumn("LOT_Delay_Code", f.expr(expr.lot_delay_code_expr)) \
        .withColumn("LOT_Delay_Code_desc", f.expr(expr.lot_delay_code_desc_exp))
    manageOutput(log, spark, step1_df, 1, "step1_df", TARGET_DB_NAME)
    log.info("Calculating destination sold to and lot delay codes has finished.")
    
    log.info("Group by load_id and count number of shipment.")
    no_of_shpmt_df = step1_df.groupBy("load_id").agg(f.countDistinct("load_id").alias("No_Of_Shipment_"))
    manageOutput(log, spark, no_of_shpmt_df, 1, "no_of_shpmt_df", TARGET_DB_NAME)
    log.info("Group by load_id and count number of shipment has finished.")
    
    max_event_datetm_dm = csot_data_group1_df.groupBy("load_id").agg(
        f.max("trnsp_stage_num").alias("max_trnsp_stage_num"),
        f.max("event_datetm").alias("max_event_datetm")
    ).withColumnRenamed("load_id", "load_id_max")
    
    max_event_datetm_join_dm = csot_data_df.join(
        max_event_datetm_dm,
        (csot_data_df.load_id == max_event_datetm_dm.load_id_max) &
        (csot_data_df.trnsp_stage_num == max_event_datetm_dm.max_trnsp_stage_num) &
        (csot_data_df.event_datetm == max_event_datetm_dm.max_event_datetm),
        how='left'
    )
    
    max_event_datetm_group_dm = max_event_datetm_join_dm.groupBy("load_id", "max_trnsp_stage_num", "max_event_datetm")\
        .agg(
            f.max("ship_to_party_code").alias("max_ship_to_party_code"),
            f.max("ship_to_party_desc").alias("max_ship_to_party_desc")
        )
    
    log.info("Joining number of shipment.")
    shpmt_join_df = step1_df.join(no_of_shpmt_df, "load_id", how='left')
    manageOutput(log, spark, shpmt_join_df, 0, "shpmt_join_df", TARGET_DB_NAME)
    log.info("Joining number of shipment has finished.")
    
    log.info("Filter on time records.")
    shpmt_on_time_df = shpmt_join_df.where(shpmt_join_df.csot_failure_reason_bucket_updated_name == 'On Time')
    shpmt_on_time_iot_df = shpmt_join_df.where(shpmt_join_df.csot_failure_reason_bucket_updated_name == 'On Time')\
        .withColumn("No_Of_Shipment_On_Time_IOT", f.expr(expr.no_of_ship_on_time_expr))
    manageOutput(log, spark, shpmt_on_time_df, 0, "shpmt_on_time_df", TARGET_DB_NAME)
    log.info("Filter on time records has finished.")
    
    log.info("Group by load_id and count number of shipment on time.")
    no_of_shpmt_on_time = shpmt_on_time_df.groupBy("load_id").agg(
        f.countDistinct("load_id").alias("No_Of_Shipment_On_Time_"),
        f.max("csot_failure_reason_bucket_updated_name").alias("csot_failure_bucket_updated_name_")
    ).withColumnRenamed("load_id", "load_id_")
    no_of_shpmt_on_time_iot = shpmt_on_time_iot_df.groupBy("load_id")\
        .agg(f.max("No_Of_Shipment_On_Time_IOT").alias("No_Of_Shipment_On_Time_IOT"))
    manageOutput(log, spark, no_of_shpmt_on_time, 0, "no_of_shpmt_on_time", TARGET_DB_NAME)
    log.info("Group by load_id and count number of shipment on time has finished.")
    
    log.info("Calculations new columns.")
    drop_live_ind_desc_filter_df = csot_data_drop_live_ind_df.where(
        (csot_data_drop_live_ind_df.drop_live_ind_desc == 'Drop') |
        (csot_data_drop_live_ind_df.drop_live_ind_desc == 'Live') |
        (csot_data_drop_live_ind_df.drop_live_ind_desc == '')
    )
    shpmt_on_time_join_df = shpmt_join_df.join(
        no_of_shpmt_on_time,
        (shpmt_join_df.load_id == no_of_shpmt_on_time.load_id_) &
        (shpmt_join_df.csot_failure_reason_bucket_updated_name == no_of_shpmt_on_time.csot_failure_bucket_updated_name_),
        how='left'
    ).join(no_of_shpmt_on_time_iot, "load_id", how='left')\
     .join(drop_live_ind_desc_filter_df, "load_id", how='left')\
     .withColumn("Lanes", f.concat(
            f.col('origin_zone_code'),
            f.lit(' to '),
            f.col('dest_city_name'),
            f.lit(','),
            f.col('dest_state_code'),
            f.lit(','),
            f.col('dest_postal_code')
        ))\
     .withColumn("otd_cnt_new", f.expr(expr.otd_cnt_expr))\
     .withColumn("TAT_Late_Counter", f.expr(expr.tat_late_cnt_expr))\
     .withColumn("FA_Flag", f.expr(expr.fa_flag_expr))\
     .withColumn("primary_carr_flag_new", f.expr(expr.prim_carr_flag_new_expr))\
     .withColumn("lot_delay_bucket", f.expr(expr.lot_delay_buck_expr))\
     .withColumn("parent_carrier_name", f.expr(expr.parent_carrier_name_expr))\
     .withColumn("lot_flag", f.expr(expr.lot_flag_div_exp))\
     .withColumn("LOT_Flag_Y_N", f.expr(expr.lot_flag_expr))\
     .withColumn("Exception", f.expr(expr.exception_expr))\
     .drop("load_id_").drop("csot_failure_bucket_updated_name_")
    manageOutput(log, spark, shpmt_on_time_join_df, 0, "shpmt_on_time_join_df", TARGET_DB_NAME)
    log.info("Calculations new columns has finished.")
    
    log.info("Step4_tab filter.")
    step4_tab_filter_df = shpmt_on_time_join_df.where(
        (shpmt_on_time_join_df.sd_doc_item_overall_process_status_val == '7') &
        (shpmt_on_time_join_df.schedule_date != 'NULL') &
        (shpmt_on_time_join_df.tender_date != 'NULL') &
        (shpmt_on_time_join_df.first_dlvry_appt_date != 'NULL')
    )
    manageOutput(log, spark, step4_tab_filter_df, 0, "step4_tab_filter_df", TARGET_DB_NAME)
    log.info("Step4_tab filter has finished.")
    
    log.info("Group by load_id and count no_of_measurable_shipments.")
    no_of_measur_shipm_df = step4_tab_filter_df.groupBy("load_id")\
        .agg(f.countDistinct("load_id").alias("no_of_measurable_shipments"))\
        .withColumn("no_of_measurable_shipments_iot", f.expr(expr.measur_ship_expr))
    manageOutput(log, spark, no_of_measur_shipm_df, 0, "no_of_measur_shipm_df", TARGET_DB_NAME)
    log.info("Group by load_id and count no_of_measurable_shipments has finished.")
    
    log.info("Joining no_of_measurable_shipments.")
    no_of_measur_shipm_join_df = shpmt_on_time_join_df.join(no_of_measur_shipm_df, "load_id", how='left')
    manageOutput(log, spark, no_of_measur_shipm_join_df, 0, "no_of_measur_shipm_join_df", TARGET_DB_NAME)
    log.info("Joining no_of_measurable_shipments has finished.")
    
    log.info("Group by load_id and count percentage_on_time.")
    percentage_on_time_df = no_of_measur_shipm_join_df.groupBy("load_id").agg(
        ((f.countDistinct("No_Of_Shipment_On_Time_IOT")/f.countDistinct("no_of_measurable_shipments_iot")) * 100).alias("percentage_on_time_iot"),
        ((f.count("No_Of_Shipment_On_Time_IOT")/f.count("no_of_measurable_shipments_iot")) * 100).alias("percentage_on_time2_iot")
    )
    manageOutput(log, spark, percentage_on_time_df, 0, "percentage_on_time_df", TARGET_DB_NAME)
    log.info("Group by load_id and count percentage_on_time has finished.")
    
    log.info("Joining percentage_on_time.")
    join_fields = [
        "pg_order_num", "load_id", "cust_po_num", "ship_to_party_code", "ship_to_party_desc",
        "csot_failure_reason_bucket_name", "csot_failure_reason_bucket_updated_name",
        "on_time_src_code", "dest_sold_to_name", "lot_delay_code"
    ]
    final_csot_tab_df = shpmt_on_time_join_df  # Base DataFrame for demonstration
    final_csot_tab_df = final_csot_tab_df.join(percentage_on_time_df, on="load_id", how="left")\
        .withColumn("percentage_on_time", f.expr(expr.percentage_on_time_expr))
    manageOutput(log, spark, final_csot_tab_df, 0, "final_csot_tab_df", TARGET_DB_NAME)
    log.info("Joining percentage_on_time has finished.")
    
    log.info("Group by load_id and count gbu.")
    gbu_code_df = origin_gbu_df.groupBy("origin_id", "parent_loc_code")\
        .agg(f.max("gbu_code").alias("gbu"))
    manageOutput(log, spark, gbu_code_df, 0, "gbu_code_df", TARGET_DB_NAME)
    log.info("Group by gbu has finished.")
    
    log.info("Joining gbu.")
    gbu_code_join_df = final_csot_tab_df.join(
        gbu_code_df,
        (final_csot_tab_df.ship_point_code == gbu_code_df.origin_id) &
        (final_csot_tab_df.origin_zone_code == gbu_code_df.parent_loc_code),
        how='left'
    ).withColumn("gbu_new", f.expr(expr.gbu_new_exp))\
     .withColumnRenamed("dest_sold_to_name", "dest_sold_to_name_csot")\
     .drop("customer_name")
    manageOutput(log, spark, gbu_code_join_df, 1, "gbu_code_join_df", TARGET_DB_NAME)
    log.info("Joining gbu has finished.")
    
    log.info("Joining channel.")
    channel_df = gbu_code_join_df.join(
        dest_channel_cust_df,
        gbu_code_join_df.dest_sold_to_name_csot == dest_channel_cust_df.dest_sold_to_name,
        how='left'
    ).withColumnRenamed("customer_name", "customer_name")\
     .withColumnRenamed("channel_name", "channel")\
     .drop("dest_sold_to_name")\
     .withColumnRenamed("dest_sold_to_name_csot", "dest_sold_to_name")\
     .withColumn("channel_new", f.expr(expr.channel_new_exp))\
     .drop("channel")\
     .withColumn("customer", f.expr(expr.customer_exp))\
     .drop("customer_name").drop("level_name").drop("country_code")
    manageOutput(log, spark, channel_df, 1, "channel_df", TARGET_DB_NAME)
    log.info("Joining channel has finished.")
    
    log.info("Joining pgp_flag.")
    pgp_flag_join_df = channel_df.join(
        ship_to_pgp_flag_df,
        channel_df.ship_to_party_code == ship_to_pgp_flag_df.ship_to_num,
        how='left'
    )
    manageOutput(log, spark, pgp_flag_join_df, 1, "pgp_flag_join_df", TARGET_DB_NAME)
    log.info("Joining pgp_flag has finished.")
    
    log.info("Joining on_time_codes_aot_reason.")
    csot_temp7_df = pgp_flag_join_df.join(
        on_time_codes_aot_reason_df,
        pgp_flag_join_df.on_time_src_desc == on_time_codes_aot_reason_df.definition_name,
        how='left'
    ).withColumnRenamed("origin_zone_code", "parent_location")\
     .withColumnRenamed("actual_shpmt_end_date", "actual_arrival_date")
    manageOutput(log, spark, csot_temp7_df, 1, "csot_temp7_df", TARGET_DB_NAME)
    log.info("Joining on_time_codes_aot_reason has finished.")
    
    log.info("Group by load_id and count min_event_datetm.")
    min_event_datetm_dm = csot_data_group1_df.groupBy("load_id", "trnsp_stage_num")\
        .agg(f.min("event_datetm").alias("min_event_datetm"))\
        .withColumnRenamed("load_id", "load_id_1")\
        .withColumnRenamed("trnsp_stage_num", "trnsp_stage_num_1")
    manageOutput(log, spark, min_event_datetm_dm, 0, "min_event_datetm_dm", TARGET_DB_NAME)
    log.info("Group by load_id and count min_event_datetm has finished.")
    
    log.info("Group by load_id and count max_datetime_dm.")
    max_datetime_dm = csot_data_group2_df.groupBy("load_id", "trnsp_stage_num", "event_datetm")\
        .agg(
            f.max("actual_dlvry_tmstp").alias("max_actl_delvry_datetime"),
            f.max("first_appt_dlvry_tmstp").alias("max_first_appointment_dlvry_datetime")
        )\
        .withColumnRenamed("load_id", "load_id_2")\
        .withColumnRenamed("trnsp_stage_num", "trnsp_stage_num_2")
    manageOutput(log, spark, max_datetime_dm, 0, "max_datetime_dm", TARGET_DB_NAME)
    log.info("Group by load_id and count max_datetime_dm has finished.")
    
    log.info("Joining cdot_on_time.")
    cdot_on_time_join_df = min_event_datetm_dm.join(
        max_datetime_dm,
        (min_event_datetm_dm.load_id_1 == max_datetime_dm.load_id_2) &
        (min_event_datetm_dm.trnsp_stage_num_1 == max_datetime_dm.trnsp_stage_num_2) &
        (min_event_datetm_dm.min_event_datetm == max_datetime_dm.event_datetm),
        how='left'
    ).withColumn("count_cdot_ontime", f.expr(expr.count_cdot_ontime_exp))
    manageOutput(log, spark, cdot_on_time_join_df, 1, "cdot_on_time_join_df", TARGET_DB_NAME)
    log.info("Joining cdot_on_time has finished.")
    
    log.info("Group by load_id and count cdot_ontime.")
    cdot_ontime_tab_df = cdot_on_time_join_df.groupBy("load_id_1")\
        .agg((f.max("count_cdot_ontime") / f.max("trnsp_stage_num_1")).alias("cdot_ontime_max"))\
        .withColumnRenamed("load_id_1", "load_id")
    manageOutput(log, spark, cdot_ontime_tab_df, 0, "cdot_ontime_tab_df", TARGET_DB_NAME)
    log.info("Group by load_id and count cdot_ontime has finished.")
    
    log.info("Joining max_cdot_ontime.")
    max_cdot_ontime_join_df = csot_temp7_df.join(cdot_ontime_tab_df, "load_id", how='left')\
        .withColumnRenamed("cdot_ontime_max", "cdot_ontime")
    manageOutput(log, spark, max_cdot_ontime_join_df, 0, "max_cdot_ontime_join_df", TARGET_DB_NAME)
    log.info("Joining max_cdot_ontime has finished.")
    
    log.info("Calculating new columns for csot.")
    csot_tab_df = max_cdot_ontime_join_df.withColumn("customer_new", f.expr(expr.customer_new_exp))\
        .withColumn("channel_final", f.expr(expr.channel_final_exp))\
        .where((max_cdot_ontime_join_df.No_Of_Shipment_ != 0) & (max_cdot_ontime_join_df.parent_location != ""))
    manageOutput(log, spark, csot_tab_df, 0, "csot_tab_df", TARGET_DB_NAME)
    log.info("Calculating new columns for csot has finished.")
    
    csot_tab_end_df = csot_tab_df.join(max_event_datetm_group_dm, "load_id", how='left')\
        .join(tfs_df, "load_id", how='left')
    
    log.info("Group csot data.")
    final_csot_tab_df = csot_tab_end_df.groupBy(
        "pg_order_num", "load_id", "cust_po_num", "ship_to_party_code", "ship_to_party_desc",
        "csot_failure_reason_bucket_name", "csot_failure_reason_bucket_updated_name",
        "on_time_src_code", "dest_sold_to_name", "lot_delay_code"
    ).agg(
        f.max("orig_request_dlvry_from_tmstp").alias("request_dlvry_from_date"),
        f.max("request_dlvry_from_datetm").alias("request_dlvry_from_datetm"),
        f.max("orig_request_dlvry_to_tmstp").alias("request_dlvry_to_date"),
        f.max("request_dlvry_to_datetm").alias("request_dlvry_to_datetm"),
        f.max("orig_request_dlvry_from_tmstp").alias("orig_request_dlvry_from_tmstp"),
        f.max("orig_request_dlvry_to_tmstp").alias("orig_request_dlvry_to_tmstp"),
        f.max("actual_arrival_datetm").alias("actual_arrival_datetm"),
        f.max("carr_num").alias("carr_num"),
        f.max("carr_desc").alias("carr_desc"),
        f.max("ship_point_at_point_of_dprtr_code").alias("origin_code"),
        f.max("ship_point_desc").alias("origin_zone_ship_from_code"),
        f.max("load_method_num").alias("load_method_num"),
        f.max("ship_cond_desc").alias("transit_mode_name"),
        f.max("order_create_date").alias("order_create_date"),
        f.max("order_create_datetm").alias("order_create_datetm"),
        f.max("schedule_date").alias("schedule_date"),
        f.max("schedule_datetm").alias("schedule_datetm"),
        f.max("tender_date").alias("tender_date"),
        f.max("tender_datetm").alias("tender_datetm"),
        f.max("first_dlvry_appt_date").alias("first_dlvry_appt_date"),
        f.max("first_dlvry_appt_datetm").alias("first_dlvry_appt_datetm"),
        f.max("last_dlvry_appt_date").alias("last_dlvry_appt_date"),
        f.max("last_dlvry_appt_datetm").alias("last_dlvry_appt_datetm"),
        f.max("final_lrdt_date").alias("final_lrdt_date"),
        f.max("final_lrdt_datetm").alias("final_lrdt_datetm"),
        f.max("actual_load_end_date").alias("actual_load_end_date"),
        f.max("actual_load_end_datetm").alias("actual_load_end_datetm"),
        f.max("actual_ship_date").alias("actual_ship_date"),
        f.max("actual_ship_datetm").alias("actual_shpmt_start_datetm"),
        f.max("measrbl_flag").alias("measrbl_flag"),
        f.max("profl_method_code").alias("profl_method_code"),
        f.max("dest_city_name").alias("dest_city_name"),
        f.max("dest_state_code").alias("dest_state_code"),
        f.max("dest_postal_code").alias("dest_postal_code"),
        f.max("country_to_code").alias("country_to_code"),
        f.max("actual_ship_month_num").alias("actual_ship_month_num"),
        f.max("ship_week_num").alias("ship_week_num"),
        f.max("cause_code").alias("cause_code"),
        f.max("on_time_src_desc").alias("on_time_src_desc"),
        f.max("true_fa_flag").alias("true_fa_flag"),
        f.max("true_frt_type_desc").alias("freight_type_val"),
        f.max("sales_org_code").alias("sales_org_code"),
        f.max("sd_doc_item_overall_process_status_val").alias("overall_status_val"),
        f.max("multi_stop_num").alias("multi_stop_num"),
        f.max("actual_service_tms_code").alias("service_tms_code"),
        f.max("lot_cust_failure_cnt").alias("lot_cust_failure_cnt"),
        f.max("pg_failure_cnt").alias("pg_failure_cnt"),
        f.max("carr_failure_cnt").alias("carr_failure_cnt"),
        f.max("others_failure_cnt").alias("others_failure_cnt"),
        f.max("tolrnc_sot_val").alias("tolrnc_sot_val"),
        f.max("lot_delay_code_desc").alias("lot_delay_code_desc"),
        f.max("No_Of_Shipment_").alias("shpmt_cnt"),
        f.max("No_Of_Shipment_On_Time_").alias("shpmt_on_time_cnt"),
        f.max("lanes").alias("lane_name"),
        f.max("otd_cnt_new").alias("otd_cnt"),
        f.max("tat_late_counter").alias("tat_late_counter_val"),
        f.max("lot_delay_bucket").alias("lot_delay_bucket_val"),
        f.max("lot_flag_y_n").alias("lot_flag"),
        f.max("fa_flag").alias("freight_auction_flag"),
        f.max("exception").alias("exception_flag"),
        f.max("parent_carrier_name").alias("parent_carr_name"),
        f.max("lot_flag").alias("lot_rate"),
        f.max("no_of_measurable_shipments").alias("measrbl_shpmt_cnt"),
        f.avg("percentage_on_time").alias("on_time_pct"),
        f.max("gbu_new").alias("gbu_code"),
        f.max("pgp_flag").alias("pgp_flag"),
        f.max("cause_desc").alias("aot_bucket_val"),
        f.max("actual_arrival_date").alias("actual_arrival_date"),
        f.max("parent_location").alias("parent_loc_code"),
        f.max("cdot_ontime").alias("cdot_ontime_cnt"),
        f.max("customer_new").alias("customer_name"),
        f.max("channel_final").alias("channel_name"),
        f.max("max_ship_to_party_code").alias("max_ship_to_party_code"),
        f.max("max_ship_to_party_desc").alias("max_ship_to_party_desc"),
        f.max("No_Of_Shipment_On_Time_IOT").alias("iot_shpmt_on_time_load_id"),
        f.avg("percentage_on_time_iot").alias("iot_on_time_pct"),
        f.avg("percentage_on_time2_iot").alias("iot_on_time2_pct"),
        f.max("no_of_measurable_shipments_iot").alias("iot_measrbl_shpmt_load_id"),
        f.max("dest_ship_from_code").alias("dest_ship_from_code"),
        f.max("distance_per_load_qty").alias("distance_per_load_num_qty"),
        f.max("first_tendered_rdd_from_datetm").alias("first_tendered_rdd_from_datetm"),
        f.max("first_tendered_rdd_to_datetm").alias("first_tendered_rdd_to_datetm"),
        f.max("drop_live_ind_desc").alias("drop_live_ind_desc")
    )
    
    # --- CSOT Actual On Time % Calculation (CSOT v3) ---
    log.info("Calculating CSOT On Time New % (CSOT v3)")
    
    def concat_to_timestamp(date_col, time_col, tmstp_format):
        as_string = f.concat(f.col(date_col), f.lit(" "), f.col(time_col))
        as_timestamp = f.unix_timestamp(as_string, tmstp_format)
        return as_timestamp
    
    time_pattern = "dd/MM/yyyy HH:mm:ss"
    on_time_data = tvb.get_on_time_data_hub_star(
        log, spark, TARGET_DB_NAME, TARGET_DB_NAME, None, None, None, csot_v3_calc=True
    )
    on_time_data = on_time_data.withColumn("first_dlvry_appt_tmstp", concat_to_timestamp("first_dlvry_appt_date", "first_dlvry_appt_datetm", time_pattern))\
        .withColumn("last_dlvry_appt_tmstp", concat_to_timestamp("last_dlvry_appt_date", "last_dlvry_appt_datetm", time_pattern))\
        .withColumn("request_dlvry_to_tmstp", concat_to_timestamp("request_dlvry_to_date", "request_dlvry_to_datetm", time_pattern))\
        .withColumn("orig_request_dlvry_to_tmstp", concat_to_timestamp("orig_request_dlvry_to_date", "orig_request_dlvry_to_datetm", time_pattern))\
        .withColumn("actual_arrival_tmstp", concat_to_timestamp("actual_shpmt_end_date", "actual_arrival_datetm", time_pattern))
    
    csot_max_load_dlvry_grp = on_time_data.groupBy("load_id", "profl_method_code").agg(
        f.max("first_dlvry_appt_tmstp").alias("first_dlvry_appt_tmstp_max"),
        f.max("last_dlvry_appt_tmstp").alias("last_dlvry_appt_tmstp_max"),
        f.max("request_dlvry_to_tmstp").alias("request_dlvry_to_tmstp_max"),
        f.max("orig_request_dlvry_to_tmstp").alias("orig_request_dlvry_to_tmstp_max"),
        f.max("actual_arrival_tmstp").alias("actual_arrival_tmstp_max"),
        f.max("pg_order_num").alias("pg_order_num")
    ).withColumn("profl_dlvry_tmstp_max",
         f.when(f.col("profl_method_code").isin(["APFD", "RDDAPF", "APF"]), f.col("first_dlvry_appt_tmstp_max"))
          .when(f.col("profl_method_code").isin(["APLD", "RDDAPL", "APL"]), f.col("last_dlvry_appt_tmstp_max"))
          .when(f.col("profl_method_code").isin(["RDD"]), f.col("request_dlvry_to_tmstp_max"))
          .when(f.col("profl_method_code").isin(["ORDD", "OTIF"]), f.col("orig_request_dlvry_to_tmstp_max"))
          .otherwise(None)
    ).withColumn("load_lvl_on_time_pct",
         f.when(f.col("profl_dlvry_tmstp_max").isNull() | f.col("actual_arrival_tmstp_max").isNull(), f.lit(None))
          .when(f.col("profl_dlvry_tmstp_max") >= f.col("actual_arrival_tmstp_max"), f.lit(1))
          .otherwise(f.lit(0))
    )
    
    csot_on_time_pct = on_time_data.join(csot_max_load_dlvry_grp, on=["load_id", "profl_method_code", "pg_order_num"], how="left")
    
    csot_on_time_pct = csot_on_time_pct.withColumn("dest_sold_to_name", f.expr(expr.dest_sold_to_name_expr))\
        .withColumn("lot_delay_code", f.expr(expr.lot_delay_code_expr))
    
    csot_on_time_pct = csot_on_time_pct.drop("request_dlvry_to_date").drop("request_dlvry_to_datetm").drop("actual_arrival_datetm")\
        .drop("profl_method_code").drop("orig_request_dlvry_to_tmstp").drop("first_dlvry_appt_date")\
        .drop("first_dlvry_appt_datetm").drop("max_ship_to_party_code").drop("max_ship_to_party_desc")\
        .drop("country_to_code").drop("sold_to_party_desc").drop("true_frt_type_desc")\
        .drop("last_dlvry_appt_date").drop("last_dlvry_appt_datetm").drop("lot_delay_reason_code")
    
    join_fields = [
        "pg_order_num", "load_id", "cust_po_num", "ship_to_party_code", "ship_to_party_desc",
        "csot_failure_reason_bucket_name", "csot_failure_reason_bucket_updated_name",
        "on_time_src_code", "dest_sold_to_name", "lot_delay_code"
    ]
    final_csot_tab_df = shpmt_on_time_join_df  # using shpmt_on_time_join_df as the base for demonstration
    final_csot_tab_df = final_csot_tab_df.join(csot_on_time_pct, on=join_fields, how="left")
    final_csot_tab_df = final_csot_tab_df.withColumn("actual_on_time_pct",
         f.when(f.col("customer_desc") == "DOLLAR GENERAL CORPOR US", f.col("load_lvl_on_time_pct"))
          .when(f.col("customer_desc") == "CVS", f.col("load_lvl_on_time_pct"))
          .otherwise(f.col("on_time_pct"))
    )
    
    final_csot_tab_df = final_csot_tab_df.drop("first_dlvry_appt_tmstp").drop("last_dlvry_appt_tmstp").drop("request_dlvry_to_tmstp")\
         .drop("first_dlvry_appt_tmstp_max").drop("last_dlvry_appt_tmstp_max").drop("request_dlvry_to_tmstp_max")\
         .drop("orig_request_dlvry_to_tmstp_max").drop("actual_arrival_tmstp_max").drop("profl_dlvry_tmstp_max")\
         .drop("actual_arrival_tmstp")
    
    final_csot_tab_df = final_csot_tab_df.dropDuplicates(join_fields)
    final_csot_tab_df = final_csot_tab_df.withColumn("iot_shpmt_on_time_load_id", f.col("iot_shpmt_on_time_load_id").cast(LongType()))\
         .withColumn("iot_measrbl_shpmt_load_id", f.col("iot_measrbl_shpmt_load_id").cast(LongType()))\
         .withColumn("distance_per_load_num_qty", f.col("distance_per_load_num_qty").cast(DecimalType(30,8)))
         
    log.info("Calculating CSOT On Time New % has finished (CSOT v3)")
    manageOutput(log, spark, final_csot_tab_df, 0, "final_csot_tab_df", TARGET_DB_NAME)
    log.info("Group csot data has finished.")
    
    return final_csot_tab_df

def load_csot_star(log):
    """
    Creates a Spark session, removes debug tables, processes CSOT data using get_csot_star,
    and writes the resulting DataFrame to the target table with overwrite mode.
    """
    spark = get_spark_session(log, 'tfx_csot', SPARK_MASTER, SPARK_CONF)
    removeDebugTables(log, spark, TARGET_DB_NAME)
    
    log.info("Started loading {}.csot_star table.".format(TARGET_DB_NAME))
    target_table_cols = spark.table(f"{TARGET_DB_NAME}.csot_star").schema.fieldNames()
    csot_star_df = get_csot_star(log, spark).select(target_table_cols)
    
    log.info("Inserting data into {}.csot_star (overwriting old data)".format(TARGET_DB_NAME))
    csot_star_df.write.insertInto(f"{TARGET_DB_NAME}.csot_star", overwrite=True)
    log.info("Loading {}.csot_star table has finished.".format(TARGET_DB_NAME))
    return

# --- MAIN FUNCTION ---

def main():
    dbutils = get_dbutils()
    spark = get_spark_session(logging.getLogger("main"), "tfx_csot", SPARK_MASTER, SPARK_CONF)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    config = Configuration.load_for_default_environment(__file__, dbutils)
    # For demonstration, we ignore the external configuration and use constants.
    
    logger.info("Starting load for CSOT star table.")
    load_csot_star(logger)

if __name__ == "__main__":
    main()
