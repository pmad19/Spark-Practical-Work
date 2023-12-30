from pyspark.sql import SparkSession
from core.jobs.pipeline import DataPipeline
import logging

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("bigdata-2023-2024-g12") \
        .enableHiveSupport() \
        .getOrCreate()
    
    log_level = spark.conf.get('spark.log_level')
    spark.sparkContext.setLogLevel(log_level.upper())

    FORMAT = "%(asctime)-15s\t%(name)s\t%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger('bd-g12')
    
    try:
        DataPipeline(spark, logger).run()
    finally:
        spark.stop()
    