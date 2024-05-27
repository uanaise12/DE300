# spark_ml_pipeline.py
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, LongType, StringType, DoubleType
from pyspark.sql.functions import col
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

class PairwiseProduct(Transformer):
    def __init__(self, inputCols, outputCols):
        self.__inputCols = inputCols
        self.__outputCols = outputCols

    def _transform(self, df):
        for cols, out_col in zip(self.__inputCols, self.__outputCols):
            df = df.withColumn(out_col, col(cols[0]) * col(cols[1]))
        return df

class SquareFeatures(Transformer):
    def __init__(self, inputCols, outputCols):
        super().__init__()
        self.inputCols = inputCols
        self.outputCols = outputCols

    def _transform(self, df):
        for inputCol, outputCol in zip(self.inputCols, self.outputCols):
            df = df.withColumn(outputCol, col(inputCol) ** 2)
        return df

def read_data(spark: SparkSession, filepath: str) -> DataFrame:
    return spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(filepath)

def pipeline(data: DataFrame, trainingData: DataFrame, testData: DataFrame):
    numeric_features = [f.name for f in data.schema.fields if isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))]
    string_features = [f.name for f in data.schema.fields if isinstance(f.dataType, StringType)]

    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}Index") for col in string_features]
    imputer_numeric = Imputer(inputCols=numeric_features, outputCols=[f"Imputed{col}" for col in numeric_features])
    squarer = SquareFeatures(inputCols=numeric_features, outputCols=[f"{col}_squared" for col in numeric_features])
    assembler = VectorAssembler(inputCols=[f"Imputed{col}" for col in numeric_features] + [f"{col}_squared" for col in numeric_features] + [f"{col}Index" for col in string_features], outputCol="features")

    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 20]).addGrid(rf.numTrees, [10, 20, 50]).build()
    crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=3)

    pipeline = Pipeline(stages=indexers + [imputer_numeric, squarer, assembler, crossval])
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)
    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    print(f"Test set area under ROC: {auc}")

    return model

def main():
    spark = SparkSession.builder.appName("Predict Heart Disease Survival").getOrCreate()
    data = read_data(spark, "s3://your-bucket/path/to/heart_disease.csv")
    trainingData, testData = data.randomSplit([0.8, 0.2], seed=7576)
    model = pipeline(data, trainingData, testData)
    spark.stop()

if __name__ == "__main__":
    main()
