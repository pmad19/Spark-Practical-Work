from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, Normalizer, VectorAssembler, UnivariateFeatureSelector
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import functions as F
from pyspark.sql import types as T
import logging


class Preprocess:
    def __init__(self, spark: SparkSession, logger: logging, df: DataFrame) -> None:
        self.spark: SparkSession = spark
        self.logger = logger
        self.df = df
        
    def run(self) -> None:
        self.logger.info('Preprocess Job: START')

        df = self.df.drop(*['CRSDepTime', 'CRSElapsedTime'])

        df = df.withColumn('DepTime', self.df['DepTime'].cast(T.StringType()))\
                    .withColumn('CRSArrTime', self.df['CRSArrTime'].cast(T.StringType()))

        numeric_columns_list = ['DepDelay', 'Distance', 'TaxiOut', 'ArrDelay']
        #null_numeric_columns_list = self.get_null_columns(self.df, numeric_columns_list)
#
        string_columns_list = ['DayOfWeek', 'UniqueCarrier', 'FlightNum', 'Origin', 'Dest', 
                               'Date', 'Season', 'DepTime', 'CRSArrTime']
        #null_string_columns_list = self.get_null_columns(self.df, string_columns_list)
#
        #imputer = Imputer().setInputCols(null_numeric_columns_list).setOutputCols(null_numeric_columns_list).setStrategy("mean")
        imputer = Imputer().setInputCols(['ArrDelay']).setOutputCols(['ArrDelay']).setStrategy("mean")
#
        df = imputer.fit(df).transform(df)
        #
        #for column in string_columns_list:
        #    mode = df.groupBy(column).count().orderBy(F.desc("count"), F.desc(column)).first()[0]
        #    df = df.na.fill({column: mode})


        index_columns = [col + "Index" for col in string_columns_list]
        indexer = StringIndexer(inputCols=string_columns_list, outputCols=index_columns)

        # OneHotEncoder
        vec_columns = [col + "Vec" for col in index_columns]
        encoder = OneHotEncoder(inputCols=index_columns, outputCols=vec_columns)

        # VectorAssembler
        num_vec_columns = numeric_columns_list + vec_columns
        assembler = VectorAssembler(inputCols=num_vec_columns, outputCol="features")

        # Normalizer
        normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)

        # Pipeline
        pipeline = Pipeline(stages=[indexer, encoder, assembler, normalizer])
        df = pipeline.fit(df).transform(df)

        # Mostrar el esquema y las primeras 10 filas del DataFrame transformado
        #df_transformed.printSchema()
        #df_transformed.show(10, truncate=False)

        #print(self.get_null_columns(df, df.columns))
        #df.show()

        #df = assembler.transform(df)

        # Configurar el selector de características para cada métrica
        selector_fpr = UnivariateFeatureSelector(
            selectionMode="fpr",
            featuresCol="features",
            labelCol="ArrDelay",
            outputCol="selectedFeatures"
        )

        selector_fdr = UnivariateFeatureSelector(
            selectionMode="fdr",
            featuresCol="features",
            labelCol="ArrDelay",
            outputCol="selectedFeatures"
        )

        selector_fwe = UnivariateFeatureSelector(
            selectionMode="fwe",
            featuresCol="features",
            labelCol="ArrDelay",
            outputCol="selectedFeatures"
        )

        # Aplicar el selector a cada conjunto de datos
        df_fpr = selector_fpr.fit(df).transform(df)
        df_fdr = selector_fdr.fit(df).transform(df)
        df_fwe = selector_fwe.fit(df).transform(df)

        # Imprimir el número de características antes y después de la selección
        vector_size = len(df.select("features").head()[0])
        print(f"Number of features without FSS: {vector_size}")

        print("Performing FSS selection - false positive rate")
        print(f"Number of features after applying false positive rate FSS: {len(df_fpr.select('selectedFeatures').head()[0])}")

        print("Performing FSS selection - false discovery rate")
        print(f"Number of features after applying false discovery rate FSS: {len(df_fdr.select('selectedFeatures').head()[0])}")

        print("Performing FSS selection - family-wise error rate")
        print(f"Number of features after applying family-wise error FSS: {len(df_fwe.select('selectedFeatures').head()[0])}")

        # Dividir los conjuntos de datos
        (trainingData_fpr, testData_fpr) = df_fpr.randomSplit([0.7, 0.3], seed=10)
        (trainingData_fdr, testData_fdr) = df_fdr.randomSplit([0.7, 0.3], seed=10)
        (trainingData_fwe, testData_fwe) = df_fwe.randomSplit([0.7, 0.3], seed=10)

        self.logger.info('Preprocess Job: END')
        return df

        
        
    def get_null_columns(self, df: DataFrame, column_list: T.List[str]) -> T.List[str]:
        df = df.select(column_list)
        null_counts = df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])
        null_counts_dict = null_counts.collect()[0].asDict()
        null_columns = [c for c, count in null_counts_dict.items() if count > 0]
        
        return null_columns
    
    