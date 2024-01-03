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

        print("=======================  TRAINING MODELS WITH ALL PARAMS  =======================")
        [train_df, test_df] = self.df.randomSplit([0.7, 0.3],10)

        self.linear_regression(train_df, test_df, "all_params")
        self.generalized_linear_regression(train_df, test_df, "all_params")
        self.decision_tree(train_df, test_df, "all_params")
        self.random_forest(train_df, test_df, "all_params")

        print("==================  TRAINING MODELS WITH UNIVARIATE FSS (FPR)  ==================")
        [train_df, test_df] = self.df_fpr.randomSplit([0.7, 0.3],10)
        
        self.linear_regression(train_df, test_df, "fpr_fss")
        self.generalized_linear_regression(train_df, test_df, "fpr_fss")
        self.decision_tree(train_df, test_df, "fpr_fss")
        self.random_forest(train_df, test_df, "fpr_fss")

        print("==================  TRAINING MODELS WITH UNIVARIATE FSS (FDR)  ==================")
        [train_df, test_df] = self.df_fdr.randomSplit([0.7, 0.3],10)
        
        self.linear_regression(train_df, test_df, "fdr_fss")
        self.generalized_linear_regression(train_df, test_df, "fdr_fss")
        self.decision_tree(train_df, test_df, "fdr_fss")
        self.random_forest(train_df, test_df, "fdr_fss")

        print("==================  TRAINING MODELS WITH UNIVARIATE FSS (FWE)  ==================")
        [train_df, test_df] = self.df_fwe.randomSplit([0.7, 0.3],10)
        
        self.linear_regression(train_df, test_df, "fwe_fss")
        self.generalized_linear_regression(train_df, test_df, "fwe_fss")
        self.decision_tree(train_df, test_df, "fwe_fss")
        self.random_forest(train_df, test_df, "fwe_fss")

        self.logger.info('Train Job: END')

    @log_method
    def linear_regression(self, train_df, test_df, name_extension):
        lr = LinearRegression(labelCol="ArrDelay", featuresCol="features", predictionCol="predictionLR", maxIter=10)
        lr_paramGrid = (ParamGridBuilder()
                        .addGrid(lr.regParam, [0.1, 0.01])
                        .addGrid(lr.elasticNetParam, [1, 0.8, 0.5])
                        .build())
        lr_evaluator_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="predictionLR", metricName="rmse")
        lr_evaluator_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="predictionLR", metricName="r2")
        lr_cv = CrossValidator(estimator=lr, 
                            evaluator=lr_evaluator_rmse, 
                            estimatorParamMaps=lr_paramGrid, 
                            numFolds=3, 
                            parallelism=3)
        lr_model = lr_cv.fit(train_df)
        self.export_model(lr_model, "lr_model_"+name_extension)
        self.print_results("LR", lr_model, test_df, lr_evaluator_rmse, lr_evaluator_r2)

    @log_method
    def generalized_linear_regression(self, train_df, test_df, name_extension):
        glr = (GeneralizedLinearRegression()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("features")
				.setPredictionCol("predictionGLR")
				.setLink("identity")
				.setFamily("gaussian")
				.setMaxIter(10))
        
        glr_paramGrid = ParamGridBuilder().addGrid(glr.regParam, [0.1, 0.01]).build()
        glr_evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionGLR").setMetricName("rmse")
        glr_evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionGLR").setMetricName("r2")
        glr_cv = CrossValidator().setEstimator(glr).setEvaluator(glr_evaluator_rmse).setEstimatorParamMaps(glr_paramGrid)\
				.setNumFolds(3).setParallelism(3)
        glr_model = glr_cv.fit(train_df)
        self.export_model(glr_model, "glr_model_"+name_extension)
        self.print_results("GLR", glr_model, test_df, glr_evaluator_rmse, glr_evaluator_r2)

    @log_method
    def decision_tree(self, train_df, test_df, name_extension):
        dtr = (DecisionTreeRegressor()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("features")
				.setPredictionCol("predictionDTR"))
        dtr_evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionDTR").setMetricName("rmse")
        dtr_evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionDTR").setMetricName("r2")
        dtr_cv = (CrossValidator()
				.setEstimator(dtr)
				.setEvaluator(dtr_evaluator_rmse)
				.setEstimatorParamMaps(ParamGridBuilder().build())
				.setNumFolds(3) 
				.setParallelism(3))
        dtr_model = dtr_cv.fit(train_df)
        self.export_model(dtr_model, "dtr_model_"+name_extension)
        self.print_results("DTR", dtr_model, test_df, dtr_evaluator_rmse, dtr_evaluator_r2)

    @log_method
    def random_forest(self, train_df, test_df, name_extension):
        rfr = (RandomForestRegressor()
				.setLabelCol("ArrDelay")
				.setFeaturesCol("features")
				.setPredictionCol("predictionRFR"))
        rfr_evaluator_rmse = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionRFR").setMetricName("rmse")
        rfr_evaluator_r2 = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("predictionRFR").setMetricName("r2")
        rfr_cv = (CrossValidator()
				.setEstimator(rfr)
				.setEvaluator(rfr_evaluator_rmse)
				.setEstimatorParamMaps(ParamGridBuilder().build())
				.setNumFolds(3) 
				.setParallelism(3))
        rfr_model = rfr_cv.fit(train_df)
        self.export_model(rfr_model, "rfr_model_"+name_extension)
        self.print_results("RFR", rfr_model, test_df, rfr_evaluator_rmse, rfr_evaluator_r2)
    
    @log_method
    def print_results(self, type: str, model: CrossValidatorModel, test_df: DataFrame, evaluator_rmse: RegressionEvaluator, evaluator_r2: RegressionEvaluator):
        print("")
        print(f"============================  {type} RESULTS  ============================")
        predictions = model.transform(test_df)
        print("ArrDelay - Prediction results")
        predictions.select("ArrDelay", ("prediction" + type)).show()
        print(f"Root Mean Squared Error ......: ${evaluator_rmse.evaluate(predictions)}")
        print(f"R-Squared ....................: ${evaluator_r2.evaluate(predictions)}")
    
    @log_method
    def export_model(self, model: CrossValidatorModel, fileName: str):
        filePath = os.environ['PATH_PROJECT'] + "/models/" + fileName
        model.write().overwrite().save(filePath)
        







            
            
