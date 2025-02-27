import logging
import joblib
import pandas as pd
from data_preprocessing import preprocess_data
from data_acquisition import load_car_evaluation_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(filepath="isolation_forest_model.joblib"):
    """
    Loads the trained Isolation Forest model from a file.

    Args:
        filepath (str): The path to the saved model.

    Returns:
        IsolationForest: The loaded model.
    """
    logging.info(f"Loading model from: {filepath}")
    model = joblib.load(filepath)
    logging.info("Model loaded.")
    return model

def detect_anomalies(model, X_test):
    """
    Detects anomalies using the trained model.

    Args:
        model (IsolationForest): The trained model.
        X_test (pd.DataFrame): The test data.

    Returns:
        pd.Series: A Series of anomaly scores.
    """
    logging.info("Detecting anomalies...")
    anomaly_scores = model.decision_function(X_test)
    logging.info("Anomalies detected.")
    return anomaly_scores

def is_anomaly(model, data_point):
    """
    Determines if a single data point is an anomaly.

    Args:
        model (IsolationForest): The trained model.
        data_point (pd.Series): A single data point.

    Returns:
        bool: True if the data point is an anomaly, False otherwise.
    """
    score = model.decision_function([data_point])
    # Isolation forest returns negative values for anomalies.
    return score < 0

if __name__ == "__main__":
    car_data = load_car_evaluation_data()
    if car_data is not None:
        X_train, X_test, y_train, y_test = preprocess_data(car_data.copy())
        model = load_model()
        anomaly_scores = detect_anomalies(model, X_test)

        # Example: Check if a specific data point is an anomaly
        example_data_point = X_test.iloc[0]
        if is_anomaly(model, example_data_point):
            logging.info("Example data point is an anomaly.")
        else:
            logging.info("Example data point is not an anomaly.")
    else:
        logging.error("Anomaly detection failed due to data loading error.")
