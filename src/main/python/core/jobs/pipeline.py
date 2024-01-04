from pyspark.sql import SparkSession
from core.jobs.extract.extract_job import Extract
from core.jobs.analysis.exploratory_analysis_job import ExploratoryAnalysis
from core.jobs.analysis.preprocess_job import Preprocess
from core.jobs.analysis.train_job import Train
from core.jobs.transform.predict_job import Predict
import time
import logging
from distutils.util import strtobool

class DataPipeline:
    def __init__(self, spark: SparkSession, logger: logging) -> None:
        self.spark = spark
        self.logger = logger
        self.active_exploratory_analysis = strtobool(spark.conf.get("spark.pipeline.active_exploratory_analysis"))
        self.active_train = strtobool(spark.conf.get("spark.pipeline.active_train_job"))

    def run(self) -> None:
        t_start = time.time()

        print()
        extract: Extract = Extract(spark=self.spark, logger=self.logger)
        df = extract.run()

        if self.active_exploratory_analysis:
            print()
            exploratory_analysis: ExploratoryAnalysis = ExploratoryAnalysis(spark=self.spark, logger=self.logger, df=df)
            exploratory_analysis.run()

        print()
        preprocess: Preprocess = Preprocess(spark=self.spark, logger=self.logger, df=df)
        df, df_fpr, df_fdr, df_fwe = preprocess.run()

        if self.active_train:
            print()
            self.logger.info(f"Number of obsv of the dataframe {df.count()}")
            self.logger.info(f"Number of obsv of the sample {df.count()*0.002}")
            df = df.sample(0.002, seed=2024)
            train: Train = Train(spark=self.spark, logger=self.logger, df=df, df_fdr=df_fdr, df_fpr=df_fpr, df_fwe=df_fwe)
            train.run()

        predict: Predict = Predict(spark=self.spark, logger=self.logger, df=df, df_fdr=df_fdr, df_fpr=df_fpr, df_fwe=df_fwe)
        predict.run()

        t_elapsed = time.time()-t_start
        t_formatted = time.strftime('%H:%M:%S', time.gmtime(t_elapsed))
        t_microseconds = str(round(t_elapsed, 4)).split('.')[1]
        self.logger.info("Total execution time: {}.{}".format(t_formatted, t_microseconds))