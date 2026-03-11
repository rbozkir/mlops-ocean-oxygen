import ensemble_bagged_trees, gradient_boosting, knn, linear_regression, neural_network, svm
import pandas as pd 
from sklearn.model_selection import train_test_split

water_data = pd.read_csv("data/water.csv")
water_data = water_data[(water_data["O2ml_L"] >= 0) & (water_data["NO3uM"] >= 0)].copy()
features = ["Depthm", "T_degC", "PO4uM", "SiO3uM", "NO2uM", "NO3uM", "Salnty"]
target = "O2ml_L"

for col in features:
    if water_data[col].isnull().any() or water_data[col].dtype == "int64":
        water_data[col] = water_data[col].astype("float64")

Xtrain, Xtest, ytrain, ytest = train_test_split(water_data[features], water_data[target], train_size=80, random_state=42)
eval_data = Xtest.copy()
eval_data["target"] = ytest

ensemble_bagged_trees.run_experiment(Xtrain, ytrain, eval_data)
gradient_boosting.run_experiment(Xtrain, ytrain, eval_data)
knn.run_experiment(Xtrain, ytrain, eval_data)
linear_regression.run_experiment(Xtrain, ytrain, eval_data)
neural_network.run_experiment(Xtrain, ytrain, eval_data)
svm.run_experiment(Xtrain, ytrain, eval_data)