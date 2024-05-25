from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import col, when, isnan, isnull, count, avg, trim, col
import os

# Constants
DATA_FOLDER = "data"
MARITAL_STATUS_BY_GENDER = [
    ["Never-married", 47.35, 41.81],
    ["Married-AF-spouse", 67.54, 68.33],
    ["Widowed", 3.58, 11.61],
    ["Divorced", 10.82, 15.09]
]
MARITAL_STATUS_BY_GENDER_COLUMNS = ["marital_status_statistics", "male", "female"]

def read_data(spark: SparkSession) -> DataFrame:
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", FloatType(), True),
        StructField("education", StringType(), True),
        StructField("education_num", FloatType(), True),
        StructField("marital_status", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("relationship", StringType(), True),
        StructField("race", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("capital_gain", FloatType(), True),
        StructField("capital_loss", FloatType(), True),
        StructField("hours_per_week", FloatType(), True),
        StructField("native_country", StringType(), True),
        StructField("income", StringType(), True)
    ])
    data = spark.read.schema(schema).option("header", "false").option("inferSchema", "false").csv(os.path.join(DATA_FOLDER,"*.csv"))
    data = data.repartition(8)

    float_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, FloatType)]
    for v in float_columns:
        data = data.withColumn(v, col(v).cast(IntegerType()))

    string_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, StringType)]
    for column in string_columns:
        data = data.withColumn(column, trim(col(column)))
    return data

def missing_values(data: DataFrame) -> DataFrame:
    missing_values = data.select([count(when(isnan(c) | isnull(c), c)).alias(c) for c in data.columns])
    data = data.dropna()
    return data

def feature_engineering(data: DataFrame) -> DataFrame:
    integer_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, IntegerType)]
    for i, col1 in enumerate(integer_columns):
        for col2 in integer_columns[i:]:
            product_col_name = f"{col1}_x_{col2}"
            data = data.withColumn(product_col_name, col(col1) * col(col2))
    return data

def join_with_US_gender(spark: SparkSession, data: DataFrame):
    us_df = spark.createDataFrame(MARITAL_STATUS_BY_GENDER, MARITAL_STATUS_BY_GENDER_COLUMNS)
    return data.join(us_df, data.marital_status == us_df.marital_status_statistics, 'outer')

def main():
    spark = SparkSession.builder.appName("Read Adult Dataset").getOrCreate()
    data = read_data(spark)
    data = missing_values(data)
    data = feature_engineering(data)
    data = join_with_US_gender(spark, data)
    data.write.format('csv').option('header', 'true').mode('overwrite').save(os.path.join(DATA_FOLDER, 'final_data.csv'))
    spark.stop()

if __name__ == "__main__":
    main()

