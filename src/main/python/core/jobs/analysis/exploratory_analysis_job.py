from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import types as T
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from core.utils.functions import log_method
import logging


class ExploratoryAnalysis:
    def __init__(self, spark: SparkSession, logger: logging, df: DataFrame) -> None:
        self.spark: SparkSession = spark
        self.logger = logger
        self.df = df
        
    def run(self) -> DataFrame:
        self.logger.info('Exploratory Analysis Job: START')
        print("")

        numeric_columns_list = [f.name for f in self.df.schema.fields if isinstance(f.dataType, T.NumericType)]
        numeric_columns_list.remove('ArrDelay')
        
        correlation_matrix =  self.calculate_correlation_matrix(self.df, numeric_columns_list)
        print("")
        print("===========================================  CORRELATION MATRIX  ===========================================")
        print(f"Columns names: {numeric_columns_list}")
        print(correlation_matrix)
        print("")

        self.logger.info('Exploratory Analysis Job: END')

    @log_method
    def calculate_correlation_matrix(self, df: DataFrame, numeric_columns: T.List[str]):
        """
        Calculates the correlation matrix for a DataFrame

        Parameters:
        - df: DataFrame to analyze.
        - numeric_columns: List of numeric columns to include in the correlation matrix.

        Returns:
        - correlation_matrix: A corrleation matrix as a Row object.
        """
        
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol='features')
        assembled_df = assembler.transform(df.select(numeric_columns))
        correlation_matrix = Correlation.corr(assembled_df, 'features').head()

        return correlation_matrix[0]
       