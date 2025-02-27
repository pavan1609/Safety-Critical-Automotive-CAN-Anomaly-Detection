import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    median_absolute_error,
)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from anomaly_detection import load_model, make_predictions
from data_preprocessing import preprocess_data
from data_acquisition import load_public_transport_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_predictions(predictions, actual, target_column):
    """
    Evaluates the predictions using various metrics and visualizations.

    Args:
        predictions (pd.DataFrame): The DataFrame with predictions.
        actual (pd.DataFrame): The DataFrame with actual values.
        target_column (str): The name of the target column.
    """
    logging.info("Evaluating predictions...")

    # Calculate metrics
    mae = mean_absolute_error(actual[target_column], predictions["Prediction"])
    rmse = mean_squared_error(actual[target_column], predictions["Prediction"], squared=False)
    r2 = r2_score(actual[target_column], predictions["Prediction"])
    mape = mean_absolute_percentage_error(actual[target_column], predictions["Prediction"])
    explained_variance = explained_variance_score(
        actual[target_column], predictions["Prediction"]
    )
    medae = median_absolute_error(actual[target_column], predictions["Prediction"])

    logging.info(f"Mean Absolute Error (MAE): {mae}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
    logging.info(f"R-squared (R2): {r2}")
    logging.info(f"Mean Absolute Percentage Error (MAPE): {mape}")
    logging.info(f"Explained Variance Score: {explained_variance}")
    logging.info(f"Median Absolute Error (MedAE): {medae}")

    # Create visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(actual[target_column], label="Actual")
    plt.plot(predictions["Prediction"], label="Prediction")
    plt.legend()
    plt.title("Actual vs. Predicted Values")
    plt.show()

    # Plot residuals
    residuals = actual[target_column] - predictions["Prediction"]
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title("Residuals")
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=20)
    plt.title("Histogram of Residuals")
    plt.show()

    # ACF and PACF plots of residuals
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, ax=axes)
    plot_pacf(residuals, ax=axes)
    plt.tight_layout()
    plt.show()

    logging.info("Evaluation completed.")


if __name__ == "__main__":
    transport_data = load_public_transport_data()
    if transport_data is not None:
        target_column = "Count"  # Replace with the actual target column name
        train_data = transport_data.iloc[:-24]  # Use all but the last 24 hours for training
        test_data = transport_data.iloc[-24:]  # Use the last 24 hours for testing
        preprocessed_train_data = preprocess_data(train_data.copy(), target_column)
        preprocessed_test_data = preprocess_data(test_data.copy(), target_column)
        model = load_model()
        predictions = make_predictions(model, preprocessed_test_data, target_column)
        evaluate_predictions(predictions, test_data, target_column)
    else:
        logging.error("Evaluation failed due to data loading error.")
