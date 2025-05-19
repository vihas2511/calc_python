'''Set of utils functions, e.g. remove/create debug tables, get spark session
'''

import datetime
import pyspark.sql.functions as f

class ConfParams(object):
    ''' ConfParams keeps configuration in a form of class '''

    def __init__(self):
        self.SPARK_MASTER = None

    def _get_dict_from_module(self, module_name):
        from importlib import import_module, reload
        import inspect
        cfg = import_module(module_name)
        reload(cfg)
        return dict([(i, j) for i, j in inspect.getmembers(cfg) if not i.startswith('_')])

    def print_config(self, logging, debug_mode_ind, debug_postfix):
        '''Print current configuration'''
        logging.info("Configuraton has the following parameters:")
        for key, value in self.__dict__.items():
            logging.info("{}: {}".format(key, value))

    @staticmethod
    def build_from_module(logging, cnf_module_name, debug_mode_ind, debug_postfix):
        '''Build parameter dict from input file'''
        self = ConfParams()
        self.__dict__.update(self._get_dict_from_module(cnf_module_name))
        logging.info("Configuration based on {} has been created".format(cnf_module_name))
        self.print_config(logging, debug_mode_ind, debug_postfix)
        return self

def manageOutput(
        logging, spark_session, data_frame, cache_ind, data_frame_name,
        target_db_name, table_location, debug_mode_ind, debug_postfix):
    ''' Manage output: create a debug table, cache table '''

    debug_postfix_new = debug_postfix + "_DEBUG"
    temporary_view_name = data_frame_name + debug_postfix_new + "_VW"
    if cache_ind == 1: #Cache table
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
            #temporary_view_name = data_frame_name + "_VW"
            data_frame.createOrReplaceTempView(temporary_view_name)
            logging.debug(
                "Temporary view {} has been created"\
                .format(temporary_view_name))

        logging.debug("Drop table if exists {}.{}"\
            .format(target_db_name, database_object_name))
        spark_session.sql("DROP TABLE IF EXISTS {}.{}"\
            .format(target_db_name, database_object_name))

        sql_stmt = '''
        CREATE TABLE {}.{} stored as parquet location "{}/{}/"AS 
        SELECT * FROM {}
        '''.format(
            target_db_name, database_object_name, table_location,
            database_object_name.upper(), temporary_view_name
            )
        logging.debug("Creating table definition in database {}"\
            .format(sql_stmt))
        spark_session.sql(sql_stmt)
        logging.debug("Data frame {} saved in database as {}"\
            .format(data_frame_name, database_object_name))


def removeDebugTables(
        logging, spark_session, target_db_name, debug_mode_ind, debug_postfix):
    ''' Remove debug tables from previous run (if they exists)'''
    debug_postfix_new = debug_postfix + "_DEBUG"
    spark_session.sql("USE {}".format(target_db_name))
    drop_tab_list = spark_session.sql("SHOW TABLES")
    logging.info("Started dropping DEBUG tables.")
    for i in drop_tab_list.collect():
        if i.tableName.upper().endswith(debug_postfix_new):
            spark_session.sql("DROP TABLE IF EXISTS {}.{}"\
                .format(i.database.upper(), i.tableName.upper()))
            logging.info("Table {}.{} has been dropped"\
                .format(i.database.upper(), i.tableName.upper()))


def add_pre_post_fix_in_list(
        logging, prefix_name, postfix_name, cols_list, exception_list,
        debug_mode_ind, debug_postfix):
    ''' Add postfix/prefix to a list of strings (columns)'''

    out_cols_list = []
    new_prefix_name = (prefix_name + "_" if prefix_name else prefix_name)\
        .lower()
    new_postfix_name = ("_" + postfix_name if postfix_name else postfix_name)\
        .lower()
    for col_name in cols_list:
        if col_name in exception_list:
            out_cols_list.append(col_name)
        else:
            out_cols_list.append(
                '{}{}{}'.format(new_prefix_name, col_name, new_postfix_name))
    return out_cols_list


def get_spark_session(logging, job_prefix_name, master, spark_config, 
    debug_mode_ind=0):
    '''Create or get current spark session'''
    import pyspark
    from datetime import datetime
    import os
    logging.debug("Started opening the spark session")
    conf = pyspark.SparkConf()
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_name = job_prefix_name + '_' + dt + '_' + str(os.getpid())
    conf.setAppName(app_name)
    conf.setMaster(master)
    #Set job properties
    logging.info("Spark session configuration: {}".format(spark_config))
    for spark_conf in spark_config:
        conf.set(spark_conf[0], spark_conf[1])
    sparkSession = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
    #Valid log levels include: ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN
    #sparkSession.sparkContext.setLogLevel("INFO" if debug_mode_ind == 0 else "INFO")
    logging.debug("Spark session has been established")
    return sparkSession

def select_star_from_table(logging, spark_session, db_name, table_name):
    ''' get all rows and cols from table '''
    logging.info("Started selecting {} from {}.".format(table_name, db_name))
    return spark_session.table('{src_db_name}.{src_table_name}'\
        .format(src_db_name=db_name, src_table_name=table_name))

def generate_date_series(start, stop, format):
    ''' 
    Args:
        start: Start date of range.
        stop: Stop date of range.
        format: Format of string showing date

    Returns:
        List of strings which contains date range 
    '''
    return [(start + datetime.timedelta(days=x)).strftime(format) for x in range(0, (stop-start).days + 1)] 


def get_ods_params(
        logging, spark_session, src_db_name, staging_location, param_group_name,
        debug_mode_ind, debug_postfix):
    ''' Get a DF with ODS params '''
    logging.info("Started selecting params from {}.param_lkp".format(src_db_name))
    parm_group_condition = ""
    if param_group_name:
        parm_group_condition = " WHERE param_group_name = '{}'".format(param_group_name)
    param_sql = """
    SELECT param_name
         , param_val
      FROM {}.param_lkp
     {} 
    """.format(src_db_name, parm_group_condition)
    logging.info("SQL for param_sql: {}".format(param_sql))
    param_df = spark_session.sql(param_sql)
    return param_df


def get_cleaned_data(
        logging, data_frame, debug_mode_ind, debug_postfix):
    ''' Cleans data (remove unneccessary spaces etc., replcae empty strings by NULLs)'''

    import pyspark.sql.functions as f
    for col in data_frame.dtypes:
        if col[1].startswith('string'):
            #If field is string then we do TRIM & replace empty strings to NULL
            data_frame = data_frame.withColumn(
                col[0],
                f.expr("TRIM({})".format(col[0]))
                ).withColumn(
                    col[0],
                    f.expr("""CASE WHEN {col} = '' THEN NULL 
                        WHEN UPPER({col}) = 'NULL' THEN NULL
                        ELSE {col} 
                        END
                        """.format(col=col[0]))
                )
    return data_frame


def get_ods_coalesced_cols(
        logging, spark_session, data_frame, column_list,
        staging_location, debug_mode_ind, debug_postfix):
    ''' Get a DF with coalesced columns from the list. The column names have to be different. 
    The first col in a list, will be first in coalesce function.'''
    import pyspark.sql.functions as f
    local_data_frame = data_frame
    logging.info("Started coalescing a list of columns")
    for col in column_list:
        local_data_frame = local_data_frame.withColumn(
            col[0],
            f.coalesce(col[0], col[1])
        )
    logging.info("Coalescing a list of columns is finished")
    return local_data_frame


def create_real_db_name(src_db_name, sap_box_name):
    """
    fill the region name into the database name

    :param src_db_name: database name, pattern: "text_{region}_text"
    :param sap_box_name: sap box name {anp, a6p, f6p, n6p, l6p}
    """
    sap_box_name_to_bd_name_component = {"anp": "ap_anp",
                                         "a6p": "ap",
                                         "f6p": "eu",
                                         "n6p": "na",
                                         "l6p": "la"
    }
    return src_db_name.format(region=sap_box_name_to_bd_name_component.get(sap_box_name))


def add_dashes_to_date(date_col):
    """
    :param date_col: data frame column containing date in String type.

    Returns:
        data frame column with date formated yyyy-MM-dd in String type
    """
    return f.concat_ws('-',
                       f.substring(date_col, 1, 4),
                       f.substring(date_col, 5, 2),
                       f.substring(date_col, 7, 2))


def get_file_list_in_dir(logging, dir_path):
  "List files in a directory & subdirectories (RECURSIVE function)"
  import os
  file_dir_list = []
  new_obj_path = dir_path
  for filename in os.listdir(dir_path):
    new_obj_path = os.path.join(dir_path, filename)

    if os.path.isfile(new_obj_path):
      file_dir_list.append((os.path.dirname(new_obj_path), new_obj_path))
    elif os.path.isdir(new_obj_path):
      file_dir_list += get_file_list_in_dir(logging, new_obj_path)
  
  return file_dir_list



def drop_local_partition(logging, spark_session, table_name, part_spec):
    ''' Drops old partitions (older than x months) in a local managed table.'''
    logging.info("Dropping partition {} in table {}".format(part_spec, table_name))
    spark_session.sql("ALTER TABLE {} DROP PARTITION ({})".format(table_name, part_spec))
    logging.info("The {} partition in table {} has been dropped".format(part_spec, table_name))
    return 0


def drop_external_partition(logging, spark_session, table_name, part_spec, dbfs_table_location_path):
    ''' Drops old partitions (older than x months) in an external table'''
    logging.info("Dropping external partition {} in table {}".format(part_spec, table_name))
    spark_session.sql("ALTER TABLE {} DROP PARTITION ({})".format(table_name, part_spec))
    import shutil
    dbfs_part_path = dbfs_table_location_path + '/' + part_spec.replace("'", "").replace(",", "/")
    shutil.rmtree(dbfs_part_path)
    logging.info("The external directory {} has been dropped".format(dbfs_part_path))
    logging.info("The {} partition in table {} has been dropped".format(part_spec, table_name))
    return 0


def archive_directory(logging, spark_session, dir_to_archive, archive_dir, archive_format):
    ''' Archive (compress & move to archive directory) a directory.'''
    logging.info("Archiving the directory: {}".format(dir_to_archive))
    import time
    archive_file_name = time.strftime("%Y%m%d_%H%M%S") + "__" + dir_to_archive.strip("/").split("/")[-1]
    logging.info("Archive file name: {}".format(archive_file_name))
    import shutil
    archived_file_path = shutil.make_archive(base_name = archive_file_name, format = archive_format, root_dir = dir_to_archive)
    moved_archived_file_path = shutil.move(archived_file_path, archive_dir)
    logging.info("Directory {} has been archived to {}".format(dir_to_archive, moved_archived_file_path))
    return 0

def get_last_update_utc_tmstp(logging, spark_session):
    ''' Calculate the last_update_utc_tmstp.'''
    logging.info("Setting up a last_update_utc_tmstp")
    time_zone = spark_session.conf.get("spark.sql.session.timeZone")
    last_update_utc_tmstp_col = f.to_utc_timestamp(f.current_timestamp(), time_zone)
    return last_update_utc_tmstp_col
    