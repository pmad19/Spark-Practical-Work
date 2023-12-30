from pyspark.sql import SparkSession
from core.jobs.extract.extract_job import Extract
from core.jobs.analysis.exploratory_analysis_job import ExploratoryAnalysis
from core.jobs.analysis.preprocess_job import Preprocess
import time
import logging


class DataPipeline:
    def __init__(self, spark: SparkSession, logger: logging) -> None:
        self.spark = spark
        self.logger = logger

    def run(self) -> None:
        t_start = time.time()

        print()
        extract: Extract = Extract(spark=self.spark, logger=self.logger)
        df = extract.run()

        print()
        exploratory_analysis: ExploratoryAnalysis = ExploratoryAnalysis(spark=self.spark, logger=self.logger, df=df)
        exploratory_analysis.run()

        print()
        preprocess: Preprocess = Preprocess(spark=self.spark, logger=self.logger, df=df)
        preprocess.run()
    
        t_elapsed = time.time()-t_start
        t_formatted = time.strftime('%H:%M:%S', time.gmtime(t_elapsed))
        t_microseconds = str(round(t_elapsed, 4)).split('.')[1]
        self.logger.info("Total execution time: {}.{}".format(t_formatted, t_microseconds))