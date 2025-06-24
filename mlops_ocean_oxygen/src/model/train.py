import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

water_data = pd.read_csv('data/water.csv')
water_data = water_data[water_data['O2ml_L'] >= 0].copy()
features = ['Depthm','T_degC', 'PO4uM', 'SiO3uM', 'NO2uM', 'NO3uM', 'Salnty']
target = 'O2ml_L'
X = water_data[features]
y = water_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=80, random_state=42)

def evaluate(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
    }

models = {
    'Ensemble (Bagged Trees)': BaggingRegressor(estimator=RandomForestRegressor(n_estimators=1, min_samples_leaf=8, random_state=42), n_estimators=30,n_jobs=-1),
    'Ensemble (Boosted Trees)': GradientBoostingRegressor(learning_rate=0.1, n_estimators=30, max_depth=3, min_samples_leaf=8),
    'Linear Regression': LinearRegression(n_jobs=-1),
    'MLP Regressor': MLPRegressor(hidden_layer_sizes=(25,), activation='relu', max_iter=1000),
    'KNN': KNeighborsRegressor(n_jobs=-1),
    'SVM': LinearSVR(),
}


results = []

cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name} with 5-Fold CV...")
    train_scores = []
    test_scores = []
    total_time = 0

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"  Fold {fold+1}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        start = time.time()
        pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        fold_time = time.time() - start
        total_time += fold_time

        train_scores.append(evaluate(y_train, train_pred))
        test_scores.append(evaluate(y_test, test_pred))

    avg_train = pd.DataFrame(train_scores).mean()
    avg_test = pd.DataFrame(test_scores).mean()

    results.append({
        'Model': name,
        'Train RMSE': avg_train['RMSE'],
        'Train R2': avg_train['R2'],
        'Train MSE': avg_train['MSE'],
        'Train MAE': avg_train['MAE'],
        'Test RMSE': avg_test['RMSE'],
        'Test R2': avg_test['R2'],
        'Test MSE': avg_test['MSE'],
        'Test MAE': avg_test['MAE'],
        'Training Time (sec)': total_time
    })

results_df = pd.DataFrame(results)
print(results_df[['Model', 'Train RMSE', 'Train R2','Train MSE','Train MAE', 'Training Time (sec)', 'Test RMSE', 'Test R2', 'Test MSE', 'Test MAE']])
