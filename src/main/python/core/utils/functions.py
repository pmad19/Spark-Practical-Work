from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.types as T
import logging
import os

logger = logging.getLogger('bigdata-2023-2024-g12')


def log_method(func):
    def wrapper(*args, **kwargs):
        logger.info(f'Start {func.__name__}')
        result = func(*args, **kwargs)
        logger.info(f'Finish {func.__name__}')
        return result
    return wrapper

@log_method
def read_csv(spark: SparkSession, inputFilePath: str, schema: T.StructType) -> DataFrame:
    inputFilePath = os.environ['PATH_PROJECT'] + "/input/" + inputFilePath
    return spark.read.csv(inputFilePath, header=True, schema=schema)