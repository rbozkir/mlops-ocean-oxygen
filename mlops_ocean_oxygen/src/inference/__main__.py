import mlflow 
import mlflow.pyfunc
import pandas as pd
import time 

MODEL_ALIASE_NAMES = [
    '@champion_svm',
    '@champion_lr',
    '@champion_knn',
    '@champion_ensemble_rf',
    '@champion_gradient_boosting',
]

water_data = pd.read_csv("data/water.csv")
water_data = water_data[water_data["O2ml_L"] >= 0].copy()
features = ["Depthm", "T_degC", "PO4uM", "SiO3uM", "NO2uM", "NO3uM", "Salnty"]
target = "O2ml_L"
X = water_data[features]
y = water_data[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=80, random_state=42)

model_name = "Oxygen Saturation"

for model_aliase in MODEL_ALIASE_NAMES:
    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_aliase}"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred_new = model.predict(X_test)