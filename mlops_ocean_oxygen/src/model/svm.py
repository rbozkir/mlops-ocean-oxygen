import mlflow
import os
import sys
import time

sys.path.append(os.path.join("."))
from utils.metrics import evaluate
from dotenv import load_dotenv
load_dotenv("../env")

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from mlflow.models import infer_signature
import pandas as pd


def run_experiment(data, features, target):
    X = data[features]
    y = data[target]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=80, random_state=42)

    model_name = "SVM"
    eval_data = Xtest.copy()
    eval_data["target"] = ytest

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name=model_name) as run:

        model = Pipeline([('scaler', StandardScaler()), ('svm', LinearSVR())])
        train_scores = []
        test_scores = []
        total_time = 0
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            print(f"  Fold {fold+1}...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            start = time.time()
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            fold_time = time.time() - start
            total_time += fold_time

            train_scores.append(evaluate(y_train, train_pred))
            test_scores.append(evaluate(y_test, test_pred))

        avg_train = pd.DataFrame(train_scores).mean()
        avg_test = pd.DataFrame(test_scores).mean()

        cv_result = {
            "CV Train RMSE": avg_train["RMSE"],
            "CV Train R2": avg_train["R2"],
            "CV Train MSE": avg_train["MSE"],
            "CV Train MAE": avg_train["MAE"],
            "CV Test RMSE": avg_test["RMSE"],
            "CV Test R2": avg_test["R2"],
            "CV Test MSE": avg_test["MSE"],
            "CV Test MAE": avg_test["MAE"],
            "CV Training Time sec": total_time,
        }

        start_time = time.time()
        model.fit(Xtrain, ytrain)
        end_time = time.time()

        signature = infer_signature(Xtrain, model.predict(Xtrain))
        model_info = mlflow.sklearn.log_model(
            model, name=model_name, signature=signature
        )

        result = mlflow.models.evaluate(
            model_info.model_uri,
            eval_data,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )

        total_time = end_time - start_time
        
        mlflow.log_metrics(
            {
                "training_time_sec": total_time,
            }
        )
        mlflow.log_metrics(cv_result)

if __name__=="__main__":
    water_data = pd.read_csv("data/water.csv")
    water_data = water_data[water_data["O2ml_L"] >= 0].copy()
    features = ["Depthm", "T_degC", "PO4uM", "SiO3uM", "NO2uM", "NO3uM", "Salnty"]
    target = "O2ml_L"

    for col in features:
        if water_data[col].isnull().any() or water_data[col].dtype == "int64":
            water_data[col] = water_data[col].astype("float64")
    
    run_experiment(water_data, features, target)