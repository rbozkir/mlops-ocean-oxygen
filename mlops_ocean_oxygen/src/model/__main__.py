import ensemble_bagged_trees, gradient_boosting, knn, linear_regression, neural_network, svm
import pandas as pd 
from sklearn.model_selection import train_test_split
from data.preprocessing import withpandas, withspark
import time 

dataset_path = "data/CalCOFI_Database_194903-202105_csv_16October2023/194903-202105_Bottle.csv"
features = ["Depthm", "T_degC", "PO4uM", "SiO3uM", "NO2uM", "NO3uM", "Salnty"]
target = "O2ml_L"

# Pandas
start_p = time.time()
train_data, eval_data = withpandas(dataset_path, features, target)
end_p = time.time()
pandas_time = end_p - start_p

# Apache Spark
start_s = time.time()
train_data, eval_data = withspark(dataset_path, features, target)
end_s = time.time()
spark_time = end_s - start_s

print(f"\n{'Preprocessing Method':<15} | {'Computation Time (second)':<20}")
print("-" * 40)
print(f"{'Pandas':<15} | {pandas_time:.4f} s")
print(f"{'Apache Spark':<15} | {spark_time:.4f} s")

Xtrain = train_data[features]
ytrain = train_data[target]

import pdb;pdb.set_trace()
ensemble_bagged_trees.run_experiment(Xtrain, ytrain, eval_data, target)
gradient_boosting.run_experiment(Xtrain, ytrain, eval_data, target)
knn.run_experiment(Xtrain, ytrain, eval_data, target)
linear_regression.run_experiment(Xtrain, ytrain, eval_data, target)
neural_network.run_experiment(Xtrain, ytrain, eval_data, target)
svm.run_experiment(Xtrain, ytrain, eval_data, target)