import sys
import logging
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pg_composite_pipelines_configuration.configuration import Configuration
from pg_tw_fa_marm_reporting.common import get_dbutils, get_spark

def manageOutput(logging, spark_session, data_frame, cache_ind, data_frame_name,
                  target_db_name, table_location, debug_mode_ind, debug_postfix):
    debug_postfix_new = debug_postfix + "_DEBUG"
    temporary_view_name = data_frame_name + debug_postfix_new + "_VW"
    if cache_ind == 1:
        temporary_view_name = data_frame_name + debug_postfix_new + "_CACHE_VW"
        data_frame.createOrReplaceTempView(temporary_view_name)
        spark_session.sql("cache table " + temporary_view_name)
        logging.info("Data frame cached as {}".format(temporary_view_name))
    elif cache_ind == 2:
        data_frame.cache()
    elif cache_ind == 3:
        from pyspark.storagelevel import StorageLevel
        data_frame.persist(StorageLevel.MEMORY_AND_DISK)

    if debug_mode_ind:
        database_object_name = data_frame_name + debug_postfix_new
        if not cache_ind:
            data_frame.createOrReplaceTempView(temporary_view_name)
            logging.debug("Temporary view {} has been created".format(temporary_view_name))

        logging.debug("Drop table if exists {}.{}".format(target_db_name, database_object_name))
        spark_session.sql("DROP TABLE IF EXISTS {}.{}".format(target_db_name, database_object_name))

        sql_stmt = '''
        CREATE TABLE {}.{} stored as parquet location "{}/{}/" AS 
        SELECT * FROM {}
        '''.format(target_db_name, database_object_name, table_location, database_object_name.upper(), temporary_view_name)
        logging.debug("Creating table definition in database {}".format(sql_stmt))
        spark_session.sql(sql_stmt)
        logging.debug("Data frame {} saved in database as {}".format(data_frame_name, database_object_name))

def get_calc_rds_table(spark_session, rds_db_name):
    return spark_session.sql(f"SELECT * FROM {rds_db_name}.prod_uom_error_report_star")

def load_prod_uom_error_report_star(spark, logger, regional_db_name, g11_db_name, target_db_name, target_table,
                                     table_2_cmpr_name, table_2_cmpr_pk_columns, table_2_cmpr_pk_columns_new,
                                     columns_2_compare, columns_2_compare_new, region):

    df = get_calc_rds_table(spark, g11_db_name)

    df = df.withColumn("region", f.lit(region))

    df.write.mode("overwrite").saveAsTable(f"{target_db_name}.{target_table}")

    logger.info("Data loaded into table: %s.%s", target_db_name, target_table)

def main():
    spark = get_spark()
    dbutils = get_dbutils()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    config = Configuration.load_for_default_environment(__file__, dbutils)

    args = sys.argv

    catalog, schema = config["catalog-name"], config["schema-name"]

    spark = get_spark()
    g11_db_name = f"{config['src-catalog-name']}.{config['g11_db_name']}"
    schema = f"{config['catalog-name']}.{config['schema-name']}"
    target_table = f"{config['tables']['prod_uom_error_report_star']}"

    print(args)
    region = args[1]
    regional_db_name = (
        f"{config['src-catalog-name']}.{config['region-schema-name'][region]}"
    )

    logger.info("Regional source table name: %s", regional_db_name)
    logger.info("g11 source db name %s", g11_db_name)
    logger.info("target_table %s", target_table)

    load_prod_uom_error_report_star(
        spark=spark,
        logger=logger,
        regional_db_name=regional_db_name,
        g11_db_name=g11_db_name,
        target_db_name=schema,
        target_table=target_table,
        table_2_cmpr_name=None,
        table_2_cmpr_pk_columns=None,
        table_2_cmpr_pk_columns_new=None,
        columns_2_compare=None,
        columns_2_compare_new=None,
        region=region,
    )

if __name__ == "__main__":
    main()
