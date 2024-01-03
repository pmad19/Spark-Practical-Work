from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.evaluation import RegressionEvaluator
from utils.functions import log_method
import logging
import os


class Predict:
    def __init__(self, spark: SparkSession, logger: logging, df: DataFrame, df_fdr: DataFrame, df_fpr: DataFrame, df_fwe: DataFrame) -> None:
        self.spark: SparkSession = spark
        self.logger = logger
        self.df = df
        self.df_fdr = df_fdr
        self.df_fpr = df_fpr
        self.df_fwe = df_fwe
        
    def run(self):
        self.logger.info('Predict Job: START')
        print("")

        print("=======================  PREDICTING MODELS WITH ALL PARAMS  =======================")
        self.predict(self.df, "LR", "all_params")
        self.predict(self.df, "GLR", "all_params")
        self.predict(self.df, "DTR", "all_params")
        self.predict(self.df, "RFR", "all_params")

        print("")
        print("==================  PREDICTING MODELS WITH UNIVARIATE FSS (FPR)  ==================")
        self.predict(self.df_fpr, "LR", "fpr_fss")
        self.predict(self.df_fpr, "GLR", "fpr_fss")
        self.predict(self.df_fpr, "DTR", "fpr_fss")
        self.predict(self.df_fpr, "RFR", "fpr_fss")

        print("")
        print("==================  PREDICTING MODELS WITH UNIVARIATE FSS (FDR)  ==================")
        self.predict(self.df_fdr, "LR", "fdr_fss")
        self.predict(self.df_fdr, "GLR", "fdr_fss")
        self.predict(self.df_fdr, "DTR", "fdr_fss")
        self.predict(self.df_fdr, "RFR", "fdr_fss")

        print("")
        print("==================  PREDICTING MODELS WITH UNIVARIATE FSS (FWE)  ==================") 
        self.predict(self.df_fwe, "LR", "fwe_fss")
        self.predict(self.df_fwe, "GLR", "fwe_fss")
        self.predict(self.df_fwe, "DTR", "fwe_fss")
        self.predict(self.df_fwe, "RFR", "fwe_fss")

        self.logger.info('Train Job: END')

    @log_method
    def predict(self, df: DataFrame, model_name: str, analysis_type: str):
        evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("prediction"+model_name.upper()).setMetricName("rmse")
        evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("prediction"+model_name.upper()).setMetricName("r2")
        model = self.load_model(model_name.lower() + "_model_" + analysis_type.lower())
        self.print_results(model_name.upper(), model, df, evaluator_rmse, evaluator_r2)
    
    def print_results(self, type: str, model: CrossValidatorModel, df: DataFrame, evaluator_rmse: RegressionEvaluator, evaluator_r2: RegressionEvaluator):
        print("")
        print(f"============================  {type} RESULTS  ============================")
        predictions = model.transform(df)
        print("ArrDelay - Prediction results")
        predictions.select("ArrDelay", ("prediction" + type)).show()
        print(f"Root Mean Squared Error ......: ${evaluator_rmse.evaluate(predictions)}")
        print(f"R-Squared ....................: ${evaluator_r2.evaluate(predictions)}")

    @log_method
    def load_model(self, fileName: str) -> CrossValidatorModel:
        filePath = os.environ['PATH_PROJECT'] + "/models/" + fileName
        if not(os.path.exists(filePath)):
            self.logger.error(f"The {fileName} model does not exist, you have to train it first or upload it")
            self.spark.stop()
            raise Exception(f"The {fileName} model does not exist")
        else:
            return CrossValidatorModel.load(filePath)
    