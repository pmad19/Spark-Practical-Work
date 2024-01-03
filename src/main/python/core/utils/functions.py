from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.types as T
import pyspark.sql.functions as F
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
def read_input_csv(spark: SparkSession, inputFilePath: str, schema: T.StructType) -> DataFrame:
    if inputFilePath == "all":
        inputFilePath = [(os.environ['PATH_PROJECT'] + "/input/" + file) for file in os.listdir(os.environ['PATH_PROJECT'] + "/input/") if file != '.gitignore']
    else:
        inputFilePath = os.environ['PATH_PROJECT'] + "/input/" + inputFilePath

    schema.add('_corrupted_records', T.StringType(), True)

    df = spark.read.format("csv")\
           .schema(schema)\
           .option("mode", "PERMISSIVE")\
           .option("columnNameOfCorruptRecord", "_corrupted_records")\
           .option("header", True)\
           .load(inputFilePath)
    
    return df
