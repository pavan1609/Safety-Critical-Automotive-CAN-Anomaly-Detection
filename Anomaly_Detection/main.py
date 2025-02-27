import logging
from data_acquisition import load_car_evaluation_data
from data_preprocessing import preprocess_data
from model_training import train_isolation_forest, save_model
from anomaly_detection import load_model, detect_anomalies
from evaluation import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Starting anomaly detection workflow...")

    # 1. Data Acquisition
    car_data = load_car_evaluation_data()
    if car_data is None:
        logging.error("Data acquisition failed. Exiting.")
        exit(1)

    # 2. Data Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(car_data.copy())

    # 3. Model Training
    model = train_isolation_forest(X_train)
    save_model(model)

    # 4. Anomaly Detection
    anomaly_scores = detect_anomalies(model, X_test)

    # 5. Evaluation
    evaluate_model(model, X_test, y_test)

    logging.info("Anomaly detection workflow completed successfully.")
