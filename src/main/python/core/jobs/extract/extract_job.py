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
        df_processed: DataFrame = self.drop_cancelled_fields(df_processed)
        df_processed: DataFrame = self.format_time_variables(df_processed)
        df_processed: DataFrame = self.create_date_time(df_processed)
        df_processed: DataFrame = self.add_season_column(df_processed)
        df_processed = df_processed.drop(*['DayofMonth', 'Year', 'Month'])
        df_processed: DataFrame = self.add_day_time_columns(df_processed)

        #df_processed = df_processed.distinct()

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
    
    def drop_cancelled_fields(self, df: DataFrame) -> DataFrame:
        return df.filter(df['Cancelled'] == 0)\
                .drop(*['CancellationCode', 'Cancelled'])
    
    def format_time_variables(self, df: DataFrame) -> DataFrame:
        return df.withColumn('DepTime', F.when(F.length(F.col('DepTime')) < 4 , F.concat(F.lit("0"), 'DepTime')).otherwise(F.col('DepTime')))\
            .withColumn('CRSDepTime', F.when(F.length(F.col('CRSDepTime')) < 4 , F.concat(F.lit("0"), 'CRSDepTime')).otherwise(F.col('CRSDepTime')))\
            .withColumn('CRSArrTime', F.when(F.length(F.col('CRSArrTime')) < 4 , F.concat(F.lit("0"), 'CRSArrTime')).otherwise(F.col('CRSArrTime')))\
            .withColumn('DepTime',  F.concat_ws(":", F.substring(F.col('DepTime'), 1, 2), F.substring(F.col('DepTime'), 3, 4)))\
            .withColumn('CRSDepTime', F.concat_ws(":", F.substring(F.col('CRSDepTime'), 1, 2), F.substring(F.col('CRSDepTime'), 3 , 4)))\
            .withColumn('CRSArrTime', F.concat_ws(":", F.substring(F.col('CRSArrTime'),1 , 2), F.substring(F.col('CRSArrTime'), 3, 4))) 
    
    def create_date_time(self, df: DataFrame) -> DataFrame:
        return df.withColumn('Date', F.concat_ws("/", F.col('Year'), F.col('Month'), F.col('DayofMonth')))
    
    def add_season_column(self, df: DataFrame) -> DataFrame:
        return df.withColumn('Season', F.when((F.col('Month') == 12) | (F.col('Month') <= 2), F.lit("Winter"))\
                             .when((F.col('Month') > 2) | (F.col('Month') <= 5), F.lit("Spring"))\
                             .when((F.col('Month') > 5) | (F.col('Month') <= 8), F.lit("Summer"))\
                             .otherwise(F.lit("Autumn")))
    
    def add_day_time_columns(self, df: DataFrame) -> DataFrame:
        # AQUI REVISAR A VER COMO FUNCIONA MEJOR EL MODELO SI CON VARIABLE DEPTIME X3 O CON VARIABLES DEPTIME CRSDEPTIME Y CRSARRTIME
        df = df.withColumn('DepTime', F.date_format(F.col('DepTime'), "HH:mm"))\
                .withColumn('CRSDepTime', F.date_format(F.col('CRSDepTime'), "HH:mm"))\
                .withColumn('CRSArrTime', F.date_format(F.col('CRSArrTime'), "HH:mm"))\
        
        df = df.withColumn('DepTime', F.when((F.col('DepTime') > "06:00") & (F.col('DepTime') <= "12:00"), "Morning")
                               .when((F.col('DepTime') > "12:00") & (F.col('DepTime') <= "17:00"), "Afternoon")
                               .when((F.col('DepTime') > "17:00") & (F.col('DepTime') <= "20:00"), "Evening")
                               .otherwise("Night"))\
                .withColumn('CRSDepTime', F.when((F.col('CRSDepTime') > "06:00") & (F.col('CRSDepTime') <= "12:00"), "Morning")
                               .when((F.col('CRSDepTime') > "12:00") & (F.col('CRSDepTime') <= "17:00"), "Afternoon")
                               .when((F.col('CRSDepTime') > "17:00") & (F.col('CRSDepTime') <= "20:00"), "Evening")
                               .otherwise("Night"))\
                .withColumn('CRSArrTime', F.when((F.col('CRSArrTime') > "06:00") & (F.col('CRSArrTime') <= "12:00"), "Morning")
                               .when((F.col('CRSArrTime') > "12:00") & (F.col('CRSArrTime') <= "17:00"), "Afternoon")
                               .when((F.col('CRSArrTime') > "17:00") & (F.col('CRSArrTime') <= "20:00"), "Evening")
                               .otherwise("Night"))
        return df
