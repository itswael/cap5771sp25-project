import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import preprocessor as pre


def refresh_model():
    train_data = pd.read_csv("output/output.csv")
    # print("Available columns:", train_data.columns.tolist())
    X, y = pre.process_features(train_data)
    # print("Feature processing done.")
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

def predict_gross_revenue(input_data, top_performing_model):
    processed_data = pre.process_data(pd.DataFrame([input_data]))
    required_features = top_performing_model.feature_names_in_
    for feature in required_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    processed_data = processed_data[required_features]
    log_prediction = top_performing_model.predict(processed_data)
    prediction = np.exp(log_prediction) - 1
    return prediction[0]


def predictor(input_data):
    """Predict gross revenue using the trained model with retry logic for fitFailedWarning"""
    max_attempts = 3
    min_diff_threshold = 0.05  # 5% threshold
    last_prediction = None

    for attempt in range(max_attempts):
        try:
            # Refresh the model
            top_performing_model = refresh_model()

            # Make prediction
            predicted_gross = predict_gross_revenue(input_data, top_performing_model)

            # If this isn't the first attempt, check if the prediction difference is small enough
            if last_prediction is not None:
                diff_ratio = abs(predicted_gross - last_prediction) / max(last_prediction, 1)

                if diff_ratio < min_diff_threshold:
                    print(f"Prediction stabilized after {attempt + 1} attempts (diff: {diff_ratio:.2%})")
                    return predicted_gross

            # Store this prediction for comparison in the next iteration
            last_prediction = predicted_gross

            # If this is the last attempt or no fit failed warning, return the prediction
            if attempt == max_attempts - 1:
                return predicted_gross

        except Exception as e:
            error_str = str(e)
            if "FitFailedWarning" in error_str:
                print(f"Fit failed on attempt {attempt + 1}, retrying...")
                if attempt == max_attempts - 1:
                    print(f"Maximum attempts reached. Using last successful prediction.")
                    if last_prediction is None:
                        raise Exception("All fitting attempts failed")
                    return last_prediction

            else:
                # For non-fit failures, raise immediately
                print(f"Prediction failed with error: {error_str}")
                raise

    # This should not be reached but just in case
    return last_prediction

# input_data = {
#     "budget": 19000000.0,
#     "votes": 927000.0,
#     "runtime": 146.0,
#     "score": 8.4,
#     "year": 1980,
#     "released": "1980-13-06",
#     "writer": "Stephen King",
#     "rating": "R",
#     "name": "The Shining",
#     "genre": "Drama",
#     "director": "Stanley Kubrick",
#     "star": "Jack Nicholson",
#     "country": "United Kingdom",
#     "company": "Warner Bros.",
# }
# top_performing_model = refresh_model()
# predicted_gross = predict_gross(input_data, top_performing_model)
# print(f"Predicted Gross Revenue: ${predicted_gross:,.2f}")