from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.model_selection import train_test_split


def withpandas(path, features, target, train_size=0.8, seed=42):
    water_data = pd.read_csv(path, encoding='latin1')
    water_data = water_data[features + [target]].copy()
    water_data.dropna(inplace=True)

    water_data = water_data[(water_data["O2ml_L"] >= 0)
                            & (water_data["NO3uM"] >= 0)].copy()

    for col in features:
        if water_data[col].isnull().any() or water_data[col].dtype == "int64":
            water_data[col] = water_data[col].astype("float64")

    train_df, test_df = train_test_split(
        water_data[features], water_data[target], train_size=train_size, random_state=seed)
    return train_df, test_df


def withspark(path, features, target, train_size=0.8, seed=42):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.option("header", "True") \
                   .option("encoding", "ISO-8859-1") \
                   .option("inferSchema", "True") \
                   .csv(path)

    all_cols = features + [target]
    df = df.select(*all_cols)

    df = df.dropna()
    df = df.filter((col("O2ml_L") >= 0) & (col("NO3uM") >= 0))

    for column_name in features:
        data_type = dict(df.dtypes)[column_name]
        if data_type in ["int", "long"]:
            df = df.withColumn(column_name, col(column_name).cast("double"))

    train_df, test_df = df.randomSplit(
        [train_size, 1.0 - train_size], seed=seed)

    return train_df.toPandas(), test_df.toPandas()
