from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.evaluation import RegressionEvaluator
from utils.functions import log_method
import logging
import os


class Train:
    def __init__(self, spark: SparkSession, logger: logging, df: DataFrame, df_fdr: DataFrame, df_fpr: DataFrame, df_fwe: DataFrame) -> None:
        self.spark: SparkSession = spark
        self.logger = logger
        self.df = df
        self.df_fdr = df_fdr
        self.df_fpr = df_fpr
        self.df_fwe = df_fwe
        
    def run(self):
        self.logger.info('Train Job: START')
        print("")

        [train_df_all_params, test_df_all_params] = self.df.randomSplit([0.7, 0.3],10)
        [train_df_fss_fpr, test_df_fss_fpr] = self.df_fpr.randomSplit([0.7, 0.3],10)
        [train_df_fss_fdr, test_df_fss_fdr] = self.df_fdr.randomSplit([0.7, 0.3],10)
        [train_df_fss_fwe, test_df_fss_fwe] = self.df_fwe.randomSplit([0.7, 0.3],10)

        lr_evaluator_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="predictionLR", metricName="rmse")
        lr_evaluator_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="predictionLR", metricName="r2")

        glr_evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionGLR").setMetricName("rmse")
        glr_evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionGLR").setMetricName("r2")

        dtr_evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionDTR").setMetricName("rmse")
        dtr_evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionDTR").setMetricName("r2")

        rfr_evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionRFR").setMetricName("rmse")
        rfr_evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionRFR").setMetricName("r2")

        print("=======================  TRAINING MODELS WITH ALL PARAMS  =======================")
        lr_predictions_all_var = self.linear_regression(train_df_all_params, test_df_all_params, "all_params", lr_evaluator_r2, lr_evaluator_rmse)
        glr_predictions_all_var = self.generalized_linear_regression(train_df_all_params, test_df_all_params, "all_params", glr_evaluator_r2, glr_evaluator_rmse)
        dtr_predictions_all_var = self.decision_tree(train_df_all_params, test_df_all_params, "all_params", dtr_evaluator_r2, dtr_evaluator_rmse)
        rfr_predictions_all_var = self.random_forest(train_df_all_params, test_df_all_params, "all_params", rfr_evaluator_r2, rfr_evaluator_rmse)

        print("==================  TRAINING MODELS WITH UNIVARIATE FSS (FPR)  ==================")        
        lr_predictions_fpr = self.linear_regression(train_df_fss_fpr, test_df_fss_fpr, "fpr_fss", lr_evaluator_r2, lr_evaluator_rmse)
        glr_predictions_fpr = self.generalized_linear_regression(train_df_fss_fpr, test_df_fss_fpr, "fpr_fss", glr_evaluator_r2, glr_evaluator_rmse)
        dtr_predictions_fpr = self.decision_tree(train_df_fss_fpr, test_df_fss_fpr, "fpr_fss", dtr_evaluator_r2, dtr_evaluator_rmse)
        rfr_predictions_fpr = self.random_forest(train_df_fss_fpr, test_df_fss_fpr, "fpr_fss", rfr_evaluator_r2, rfr_evaluator_rmse)

        print("==================  TRAINING MODELS WITH UNIVARIATE FSS (FDR)  ==================")        
        lr_predictions_fdr = self.linear_regression(train_df_fss_fdr, test_df_fss_fdr, "fdr_fss", lr_evaluator_r2, lr_evaluator_rmse)
        glr_predictions_fdr = self.generalized_linear_regression(train_df_fss_fdr, test_df_fss_fdr, "fdr_fss", glr_evaluator_r2, glr_evaluator_rmse)
        dtr_predictions_fdr = self.decision_tree(train_df_fss_fdr, test_df_fss_fdr, "fdr_fss", dtr_evaluator_r2, dtr_evaluator_rmse)
        rfr_predictions_fdr = self.random_forest(train_df_fss_fdr, test_df_fss_fdr, "fdr_fss",  rfr_evaluator_r2, rfr_evaluator_rmse)

        print("==================  TRAINING MODELS WITH UNIVARIATE FSS (FWE)  ==================")        
        lr_predictions_fwe = self.linear_regression(train_df_fss_fwe, test_df_fss_fwe, "fwe_fss", lr_evaluator_r2, lr_evaluator_rmse)
        glr_predictions_fwe = self.generalized_linear_regression(train_df_fss_fwe, test_df_fss_fwe, "fwe_fss", glr_evaluator_r2, glr_evaluator_rmse)
        dtr_predictions_fwe = self.decision_tree(train_df_fss_fwe, test_df_fss_fwe, "fwe_fss", dtr_evaluator_r2, dtr_evaluator_rmse)
        rfr_predictions_fwe = self.random_forest(train_df_fss_fwe, test_df_fss_fwe, "fwe_fss", rfr_evaluator_r2, rfr_evaluator_rmse)

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
						("RANDOM FOREST REGRESSION - FWE FSS", rfr_predictions_fwe[0], rfr_predictions_fwe[1])]

        summary_df = self.spark.createDataFrame(summary, ["Model",'rmse', "r2"])
        summary_df.show()

        self.logger.info('Train Job: END')

    @log_method
    def linear_regression(self, train_df, test_df, name_extension, lr_evaluator_r2, lr_evaluator_rmse):
        lr = LinearRegression(labelCol="ArrDelay", featuresCol="features", predictionCol="predictionLR", maxIter=10)
        lr_paramGrid = (ParamGridBuilder()
                        .addGrid(lr.regParam, [0.1, 0.01])
                        .addGrid(lr.elasticNetParam, [1, 0.8, 0.5])
                        .build())
        lr_cv = CrossValidator(estimator=lr, 
                            evaluator=lr_evaluator_rmse, 
                            estimatorParamMaps=lr_paramGrid, 
                            numFolds=3, 
                            parallelism=3)
        lr_model = lr_cv.fit(train_df)
        self.export_model(lr_model, "lr_model_"+name_extension)
        return self.print_results("LR", lr_model, test_df, lr_evaluator_rmse, lr_evaluator_r2)

    @log_method
    def generalized_linear_regression(self, train_df, test_df, name_extension, glr_evaluator_r2, glr_evaluator_rmse):
        glr = (GeneralizedLinearRegression()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("features")
				.setPredictionCol("predictionGLR")
				.setLink("identity")
				.setFamily("gaussian")
				.setMaxIter(10))
        glr_paramGrid = ParamGridBuilder().addGrid(glr.regParam, [0.1, 0.01]).build()
        glr_cv = CrossValidator().setEstimator(glr).setEvaluator(glr_evaluator_rmse).setEstimatorParamMaps(glr_paramGrid)\
				.setNumFolds(3).setParallelism(3)
        glr_model = glr_cv.fit(train_df)
        self.export_model(glr_model, "glr_model_"+name_extension)
        return self.print_results("GLR", glr_model, test_df, glr_evaluator_rmse, glr_evaluator_r2)
        

    @log_method
    def decision_tree(self, train_df, test_df, name_extension, dtr_evaluator_r2, dtr_evaluator_rmse):
        dtr = (DecisionTreeRegressor()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("features")
				.setPredictionCol("predictionDTR"))
        dtr_cv = (CrossValidator()
				.setEstimator(dtr)
				.setEvaluator(dtr_evaluator_rmse)
				.setEstimatorParamMaps(ParamGridBuilder().build())
				.setNumFolds(3) 
				.setParallelism(3))
        dtr_model = dtr_cv.fit(train_df)
        self.export_model(dtr_model, "dtr_model_"+name_extension)
        return self.print_results("DTR", dtr_model, test_df, dtr_evaluator_rmse, dtr_evaluator_r2)

    @log_method
    def random_forest(self, train_df, test_df, name_extension, rfr_evaluator_r2, rfr_evaluator_rmse):
        rfr = (RandomForestRegressor()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("features")
				.setPredictionCol("predictionRFR"))
        rfr_cv = (CrossValidator()
				.setEstimator(rfr)
				.setEvaluator(rfr_evaluator_rmse)
				.setEstimatorParamMaps(ParamGridBuilder().build())
				.setNumFolds(3) 
				.setParallelism(3))
        rfr_model = rfr_cv.fit(train_df)
        self.export_model(rfr_model, "rfr_model_"+name_extension)
        return self.print_results("RFR", rfr_model, test_df, rfr_evaluator_rmse, rfr_evaluator_r2)
    
    @log_method
    def print_results(self, type: str, model: CrossValidatorModel, test_df: DataFrame, evaluator_rmse: RegressionEvaluator, evaluator_r2: RegressionEvaluator):
        print("")
        print(f"============================  {type} RESULTS  ============================")
        predictions = model.transform(test_df)
        print("ArrDelay - Prediction results")
        predictions.select("ArrDelay", ("prediction" + type)).show()
        rmse =  evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)
        print(f"Root Mean Squared Error ......: ${rmse}")
        print(f"R-Squared ....................: ${r2}")
        return rmse, r2
    
    @log_method
    def export_model(self, model: CrossValidatorModel, fileName: str):
        filePath = os.environ['PATH_PROJECT'] + "/models/" + fileName
        model.write().overwrite().save(filePath)
        







            
            
