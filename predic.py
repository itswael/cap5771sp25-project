import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import preprocessor as pre


def refresh_model():
    train_data = pd.read_csv("output/output.csv")
    # print("Available columns:", df.columns.tolist())
    X, y = pre.process_features(train_data)
    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }
    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    top_parameters = grid_search.best_params_
    top_performing_model = XGBRegressor(
        objective="reg:squarederror", random_state=42, **top_parameters
    )
    top_performing_model.fit(X, y)
    return top_performing_model

