from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, Normalizer, VectorAssembler, UnivariateFeatureSelector, UnivariateFeatureSelectorModel
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.sql import functions as F
from pyspark.sql import types as T
from utils.functions import log_method
from distutils.util import strtobool
import logging
import os


class Preprocess:
    def __init__(self, spark: SparkSession, logger: logging, df: DataFrame) -> None:
        self.spark: SparkSession = spark
        self.logger = logger
        self.df = df
        self.with_preprocess_training = strtobool(spark.conf.get('spark.pipeline.active_preprocess_job'))
        self.with_fpr_fss = strtobool(spark.conf.get('spark.pipeline.active_fpr_fss'))
        self.with_fdr_fss = strtobool(spark.conf.get('spark.pipeline.active_fdr_fss'))
        self.with_fwe_fss = strtobool(spark.conf.get('spark.pipeline.active_fwe_fss'))
        
    def run(self):
        self.logger.info('Preprocess Job: START')
        print("")

        df = self.df.drop(*['CRSDepTime', 'CRSElapsedTime'])

        df: DataFrame = self.cast_columns_types(df)
        df: DataFrame = self.get_features_dataframe(df)

        filePath = os.environ['PATH_PROJECT'] + "/models/univariate_filters"
        
        df_fpr = self.apply_univariate_fss_fpr(filePath, df)
        df_fdr = self.apply_univariate_fss_fdr(filePath, df)
        df_fwe = self.apply_univariate_fss_fwe(filePath, df)

        self.print_fss_summary(df, df_fpr, df_fdr, df_fwe)

        self.logger.info('Preprocess Job: END')
        return df, df_fpr, df_fdr, df_fwe

        
    @log_method
    def cast_columns_types(self, df: DataFrame) -> DataFrame:
        return df.withColumn('DepTime', self.df['DepTime'].cast(T.StringType()))\
                    .withColumn('CRSArrTime', self.df['CRSArrTime'].cast(T.StringType()))
    
    @log_method
    def get_features_dataframe(self, df: DataFrame) -> DataFrame:
        numeric_columns_list = ['DepDelay', 'Distance', 'TaxiOut']
        string_columns_list = ['DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest', 
                               'Date', 'Season', 'DepTime', 'CRSArrTime']

        if self.with_preprocess_training:
            # Impute null values
            imputer = Imputer().setInputCols(['ArrDelay']).setOutputCols(['ArrDelay']).setStrategy("mean")

            # Index categorical variables
            index_columns = [col + "_index" for col in string_columns_list]
            indexer = StringIndexer(inputCols=string_columns_list, outputCols=index_columns, handleInvalid="keep")

            # OneHotEncoder
            vec_columns = [col + "_vec" for col in index_columns]
            encoder = OneHotEncoder(inputCols=index_columns, outputCols=vec_columns)

            # VectorAssembler
            num_vec_columns = numeric_columns_list + vec_columns
            assembler = VectorAssembler(inputCols=num_vec_columns, outputCol="features")

            # Normalize data
            normalizer = Normalizer(inputCol='features', outputCol='normfeatures', p=1.0)

            # Pipeline
            pipeline = Pipeline(stages=[imputer, indexer, encoder, assembler, normalizer])
            pipeline_trained = pipeline.fit(df)
            pipeline_trained.write().overwrite().save(os.environ['PATH_PROJECT'] + "/models/preprocess_pipeline")
        else: 
            if not(os.path.exists(os.environ['PATH_PROJECT'] + "/models/preprocess_pipeline")):
                self.logger.error("The preprocess pipeline file does not exist, you have to train it first or upload it")
                self.spark.stop()
                raise Exception("The preprocess pipeline file does not exist")
            else: 
                pipeline_trained = PipelineModel.load(os.environ['PATH_PROJECT'] + "/models/preprocess_pipeline")
                

        df = pipeline_trained.transform(df)
    
        return df.select(['ArrDelay', 'normfeatures']).withColumnRenamed('normfeatures', 'features')

    @log_method
    def get_univariate_fss_fpr(self, df: DataFrame) -> UnivariateFeatureSelectorModel:
        selector_fpr = UnivariateFeatureSelector(
            selectionMode="fpr",
            labelCol="ArrDelay",
            outputCol="selectedFeatures"
        ).setFeatureType("continuous").setLabelType("continuous")\
        .setSelectionThreshold(0.05)
        return selector_fpr.fit(df)
    
    @log_method
    def get_univariate_fss_fdr(self, df: DataFrame) -> UnivariateFeatureSelectorModel:
        selector_fpr = UnivariateFeatureSelector(
            selectionMode="fdr",
            labelCol="ArrDelay",
            outputCol="selectedFeatures"
        ).setFeatureType("continuous").setLabelType("continuous")\
        .setSelectionThreshold(0.05)
        return selector_fpr.fit(df)
    
    @log_method
    def get_univariate_fss_fwe(self, df: DataFrame) -> UnivariateFeatureSelectorModel:
        selector_fpr = UnivariateFeatureSelector(
            selectionMode="fwe",
            labelCol="ArrDelay",
            outputCol="selectedFeatures"
        ).setFeatureType("continuous").setLabelType("continuous")\
        .setSelectionThreshold(0.05)
        return selector_fpr.fit(df)
    
    @log_method
    def apply_univariate_fss_fpr(self, filePath: str, df: DataFrame) -> DataFrame:
        if self.with_fpr_fss:
            univariate_fss_fpr: UnivariateFeatureSelectorModel =self.get_univariate_fss_fpr(df)
            univariate_fss_fpr.write().overwrite().save(filePath + '/univariate_fss_fpr')
        else:
            if not(os.path.exists(filePath + '/univariate_fss_fpr')):
                self.logger.error("The univariate_fss_fpr model does not exist, you have to train it first or upload it")
                self.spark.stop()
                raise Exception("The univariate_fss_fpr model does not exist")
            else: 
                univariate_fss_fpr = UnivariateFeatureSelectorModel.load(filePath + '/univariate_fss_fpr')
        return univariate_fss_fpr.transform(df)
    
    @log_method
    def apply_univariate_fss_fdr(self, filePath: str, df: DataFrame) -> DataFrame:
        if self.with_fdr_fss:
            univariate_fss_fdr: UnivariateFeatureSelectorModel = self.get_univariate_fss_fdr(df)
            univariate_fss_fdr.write().overwrite().save(filePath + '/univariate_fss_fdr')
        else:
            if not(os.path.exists(filePath + '/univariate_fss_fdr')):
                self.logger.error("The univariate_fss_fdr model does not exist, you have to train it first or upload it")
                self.spark.stop()
                raise Exception("The univariate_fss_fdr model does not exist")
            else: 
                univariate_fss_fdr = UnivariateFeatureSelectorModel.load(filePath + '/univariate_fss_fdr')
        return univariate_fss_fdr.transform(df)
    
    @log_method
    def apply_univariate_fss_fwe(self, filePath: str, df: DataFrame) -> DataFrame:
        if self.with_fwe_fss:
            univariate_fss_fwe: UnivariateFeatureSelectorModel = self.get_univariate_fss_fwe(df)
            univariate_fss_fwe.write().overwrite().save(filePath + '/univariate_fss_fwe')
        else:
            if not(os.path.exists(filePath + '/univariate_fss_fwe')):
                self.logger.error("The univariate_fss_fwe model does not exist, you have to train it first or upload it")
                self.spark.stop()
                raise Exception("The univariate_fss_fwe model does not exist")
            else: 
                univariate_fss_fwe = UnivariateFeatureSelectorModel.load(filePath + '/univariate_fss_fwe')
        return univariate_fss_fwe.transform(df)

    def print_fss_summary(self, df:DataFrame, df_fpr: DataFrame, df_fdr: DataFrame, df_fwe: DataFrame) -> None:
        vector_size = len(df.select("features").head()[0])
        print("")
        print("============================  FSS SUMMARY  ============================")
        print(f"Number of features without FSS....................................: {vector_size}")
        if df_fpr: print(f"Number of features after applying false positive rate (fpr) FSS...: {len(df_fpr.select('selectedFeatures').head()[0])}")
        if df_fdr: print(f"Number of features after applying false discovery rate (fdr) FSS..: {len(df_fdr.select('selectedFeatures').head()[0])}")
        if df_fwe: print(f"Number of features after applying family-wise error (fwe) FSS.....: {len(df_fwe.select('selectedFeatures').head()[0])}")
        print("")        
        
        
