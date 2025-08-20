import mlflow 
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
import pandas as pd
import time 
from dotenv import load_dotenv
load_dotenv("../env")

MODEL_ALIASE_NAMES = [
    'champion_ensemble_rf',
    'champion_gradient_boosting',
    'champion_knn',
    'champion_lr',
    'champion_nn',
    'champion_svm',
]

water_data = pd.read_csv("data/water.csv")
water_data = water_data[water_data["O2ml_L"] >= 0].copy()
features = ["Depthm", "T_degC", "PO4uM", "SiO3uM", "NO2uM", "NO3uM", "Salnty"]
target = "O2ml_L"
X = water_data[features]
y = water_data[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=80, random_state=42)
test_data_lenght = len(Xtest)
model_name = "Oxygen Saturation"

all_time_analysis = []

for model_aliase in MODEL_ALIASE_NAMES:
    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}@{model_aliase}"
    model = mlflow.sklearn.load_model(model_uri)

    # Model Inference
    start = time.time()
    y_pred_new = model.predict(Xtest)
    pred_time = time.time() - start

    # Calculate the average of Inference Time for test data 
    inference_time_avg = pred_time / test_data_lenght
    
    time_analysis = {
        "model name": model_name,
        "model aliase": model_aliase,
        "test data lenght": test_data_lenght, 
        "inference time second (total) ": "{:.3f}".format(pred_time),
        "inference time microsecond (avg) ": "{:.3f}".format(inference_time_avg * 10**6),
    }

    all_time_analysis.append(time_analysis)
    
results = pd.DataFrame.from_dict(all_time_analysis)
results.to_csv('../reports/inference_time_analysis.csv', index=False)  


    