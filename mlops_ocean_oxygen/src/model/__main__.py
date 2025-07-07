import ensemble_bagged_trees, gradient_boosting, knn, linear_regression, neural_network, svm
import pandas as pd 

water_data = pd.read_csv("data/water.csv")
water_data = water_data[(water_data["O2ml_L"] >= 0) & (water_data["NO3uM"] >= 0)].copy()
features = ["Depthm", "T_degC", "PO4uM", "SiO3uM", "NO2uM", "NO3uM", "Salnty"]
target = "O2ml_L"

for col in features:
    if water_data[col].isnull().any() or water_data[col].dtype == "int64":
        water_data[col] = water_data[col].astype("float64")

ensemble_bagged_trees.run_experiment(water_data, features, target)
gradient_boosting.run_experiment(water_data, features, target)
knn.run_experiment(water_data, features, target)
linear_regression.run_experiment(water_data, features, target)
neural_network.run_experiment(water_data, features, target)
svm.run_experiment(water_data, features, target)