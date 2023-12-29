from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
import os
import logging


class Extract:
    def __init__(self, spark: SparkSession, logger: logging) -> None:
        self.spark: SparkSession = spark
        self.logger = logger
        self.inputFilePath =  os.environ['PATH_PROJECT'] + "/input/" + \
                            self.spark.conf.get('spark.data.input_file')
        
    def run(self) -> None:
        self.logger.info('Extract Job: START')
        schema: T.StructType = self.get_schema()
        df: DataFrame = self.spark.read.csv(self.inputFilePath, header=True, schema=schema)
        df_processed: DataFrame = self.drop_forbidden_variables(df)
        
        df_processed = df_processed.filter(df_processed['Cancelled'] == 0)

        df_processed = df_processed.withColumn('DepTime', F.when(F.length(F.col('DepTime')) < 4 , F.concat(F.lit("0"), 'DepTime')).otherwise(F.col('DepTime')))\
            .withColumn('CRSDepTime', F.when(F.length(F.col('CRSDepTime')) < 4 , F.concat(F.lit("0"), 'CRSDepTime')).otherwise(F.col('CRSDepTime')))\
            .withColumn('CRSArrTime', F.when(F.length(F.col('CRSArrTime')) < 4 , F.concat(F.lit("0"), 'CRSArrTime')).otherwise(F.col('CRSArrTime')))\
            .withColumn('DepTime',  F.concat_ws(":", F.substring(F.col('DepTime'), 1, 2), F.substring(F.col('DepTime'), 3, 4)))\
            .withColumn('CRSDepTime', F.concat_ws(":", F.substring(F.col('CRSDepTime'), 1, 2), F.substring(F.col('CRSDepTime'), 3 , 4)))\
            .withColumn('CRSArrTime', F.concat_ws(":", F.substring(F.col('CRSArrTime'),1 , 2), F.substring(F.col('CRSArrTime'), 3, 4))) 
        
        df_processed.show()
        self.logger.info('Extract Job: END')
    
    def get_schema(self) -> T.StructType:
        """
        Get data schema
        Returns:
            A struct with the schema
        """
        return T.StructType([
            T.StructField("Year", T.IntegerType(), True),
            T.StructField("Month", T.IntegerType(), True),
            T.StructField("DayofMonth", T.IntegerType(), True),
            T.StructField("DayOfWeek", T.StringType(), True),
            T.StructField("DepTime", T.StringType(), True),
            T.StructField("CRSDepTime", T.StringType(), True),
            T.StructField("ArrTime", T.IntegerType(), True),
            T.StructField("CRSArrTime", T.StringType(), True),
            T.StructField("UniqueCarrier", T.StringType(), True),
            T.StructField("FlightNum", T.IntegerType(), True),
            T.StructField("TailNum", T.StringType(), True),
            T.StructField("ActualElapsedTime", T.IntegerType(), True),
            T.StructField("CRSElapsedTime", T.IntegerType(), True),
            T.StructField("AirTime", T.IntegerType(), True),
            T.StructField("ArrDelay", T.IntegerType(), True),
            T.StructField("DepDelay", T.IntegerType(), True),
            T.StructField("Origin", T.StringType(), True),
            T.StructField("Dest", T.StringType(), True),
            T.StructField("Distance", T.IntegerType(), True),
            T.StructField("TaxiIn", T.IntegerType(), True),
            T.StructField("TaxiOut", T.IntegerType(), True),
            T.StructField("Cancelled", T.IntegerType(), True),
            T.StructField("CancellationCode", T.StringType(), True),
            T.StructField("Diverted", T.IntegerType(), True),
            T.StructField("CarrierDelay", T.IntegerType(), True),
            T.StructField("WeatherDelay", T.IntegerType(), True),
            T.StructField("NASDelay", T.IntegerType(), True),
            T.StructField("SecurityDelay", T.IntegerType(), True),
            T.StructField("LateAircraftDelay", T.IntegerType(), True),
        ])
    
    def drop_forbidden_variables(self, df: DataFrame) -> DataFrame:
        return df.drop(*['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn',
                                'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
                                'SecurityDelay', 'LateAircraftDelay'])