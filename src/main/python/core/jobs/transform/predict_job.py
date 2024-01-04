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
        lr_predictions_all_var = self.predict(self.df, "LR", "all_params")
        glr_predictions_all_var = self.predict(self.df, "GLR", "all_params")
        dtr_predictions_all_var = self.predict(self.df, "DTR", "all_params")
        rfr_predictions_all_var = self.predict(self.df, "RFR", "all_params")

        print("")
        print("==================  PREDICTING MODELS WITH UNIVARIATE FSS (FPR)  ==================")
        lr_predictions_fpr = self.predict(self.df_fpr, "LR", "fpr_fss")
        glr_predictions_fpr = self.predict(self.df_fpr, "GLR", "fpr_fss")
        dtr_predictions_fpr = self.predict(self.df_fpr, "DTR", "fpr_fss")
        rfr_predictions_fpr = self.predict(self.df_fpr, "RFR", "fpr_fss")

        print("")
        print("==================  PREDICTING MODELS WITH UNIVARIATE FSS (FDR)  ==================")
        lr_predictions_fdr = self.predict(self.df_fdr, "LR", "fdr_fss")
        glr_predictions_fdr = self.predict(self.df_fdr, "GLR", "fdr_fss")
        dtr_predictions_fdr = self.predict(self.df_fdr, "DTR", "fdr_fss")
        rfr_predictions_fdr = self.predict(self.df_fdr, "RFR", "fdr_fss")

        print("")
        print("==================  PREDICTING MODELS WITH UNIVARIATE FSS (FWE)  ==================") 
        lr_predictions_fwe = self.predict(self.df_fwe, "LR", "fwe_fss")
        glr_predictions_fwe = self.predict(self.df_fwe, "GLR", "fwe_fss")
        dtr_predictions_fwe = self.predict(self.df_fwe, "DTR", "fwe_fss")
        rfr_predictions_fwe = self.predict(self.df_fwe, "RFR", "fwe_fss")

        print("==================  SUMMARY  ==================")
        summary= [
            ("LINEAR REGRESSION - ALL VARIABLES", lr_predictions_all_var[0], lr_predictions_all_var[1]),
			("LINEAR REGRESSION - FPR FSS", lr_predictions_fpr[0], lr_predictions_fpr[1]),
			("LINEAR REGRESSION - FDR FSS", lr_predictions_fdr[0], lr_predictions_fdr[1]),
			("LINEAR REGRESSION - FWE FSS", lr_predictions_fwe[0], lr_predictions_fwe[1]),
            ("GENERALIZED LINEAR REGRESSION - ALL VARIABLES", glr_predictions_all_var[0], glr_predictions_all_var[1]),
			("GENERALIZED LINEAR REGRESSION - FPR FSS", glr_predictions_fpr[0], glr_predictions_fpr[1]),
            ("GENERALIZED LINEAR REGRESSION - FDR FSS", glr_predictions_fdr[0], glr_predictions_fdr[1]),
            ("GENERALIZED LINEAR REGRESSION - FWE FSS", glr_predictions_fwe[0], glr_predictions_fwe[1]),
			("DECISION TREE REGRESSION - ALL VARIABLES", dtr_predictions_all_var[0], dtr_predictions_all_var[1]),
            ("DECISION TREE REGRESSION - FPR FSS", dtr_predictions_fpr[0], dtr_predictions_fpr[1]),
			("DECISION TREE REGRESSION - FDR FSS", dtr_predictions_fdr[0], dtr_predictions_fdr[1]),
			("DECISION TREE REGRESSION - FWE FSS", dtr_predictions_fwe[0], dtr_predictions_fwe[1]),
            ("RANDOM FOREST REGRESSION - ALL VARIABLES", rfr_predictions_all_var[0], rfr_predictions_all_var[1]),
			("RANDOM FOREST REGRESSION - FPR FSS", rfr_predictions_fpr[0], rfr_predictions_fpr[1]),
			("RANDOM FOREST REGRESSION - FDR FSS", rfr_predictions_fdr[0], rfr_predictions_fdr[1]),
			("RANDOM FOREST REGRESSION - FWE FSS", rfr_predictions_fwe[0], rfr_predictions_fwe[1])
            ]

        summary_df = self.spark.createDataFrame(summary, ["Model",'rmse', "r2"])
        summary_df.show()

        self.logger.info('Train Job: END')

    @log_method
    def predict(self, df: DataFrame, model_name: str, analysis_type: str) -> [float, float]:
        evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("prediction"+model_name.upper()).setMetricName("rmse")
        evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("prediction"+model_name.upper()).setMetricName("r2")
        model = self.load_model(model_name.lower() + "_model_" + analysis_type.lower())
        return self.print_results(model_name.upper(), model, df, evaluator_rmse, evaluator_r2)
    
    def print_results(self, type: str, model: CrossValidatorModel, df: DataFrame, evaluator_rmse: RegressionEvaluator, evaluator_r2: RegressionEvaluator) -> [float, float]:
        print("")
        print(f"============================  {type} RESULTS  ============================")
        predictions = model.transform(df)
        print("ArrDelay - Prediction results")
        predictions.select("ArrDelay", ("prediction" + type)).show()
        r2 = evaluator_r2.evaluate(predictions)
        rmse = evaluator_rmse.evaluate(predictions)
        print(f"Root Mean Squared Error ......: {rmse}")
        print(f"R-Squared ....................: {r2}")
        return rmse, r2

    @log_method
    def load_model(self, fileName: str) -> CrossValidatorModel:
        filePath = os.environ['PATH_PROJECT'] + "/models/" + fileName
        if not(os.path.exists(filePath)):
            self.logger.error(f"The {fileName} model does not exist, you have to train it first or upload it")
            self.spark.stop()
            raise Exception(f"The {fileName} model does not exist")
        else:
            return CrossValidatorModel.load(filePath)
    