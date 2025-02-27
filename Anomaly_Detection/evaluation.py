import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from anomaly_detection import load_model, detect_anomalies
from data_preprocessing import preprocess_data
from data_acquisition import load_car_evaluation_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance using classification metrics and visualizations.

    Args:
        model: The trained anomaly detection model.
        X_test: The test data features.
        y_test: The true labels for the test data.
    """
    logging.info("Evaluating model...")
    anomaly_scores = detect_anomalies(model, X_test)
    # Convert anomaly scores to binary predictions (anomalies vs. normal)
    predictions = (anomaly_scores < 0).astype(int)

    # Print classification report and confusion matrix
    logging.info("Classification Report:\n" + classification_report(y_test, predictions))
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, predictions)))

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, -anomaly_scores) #negative anomaly scores, because ROC curve expects higher scores for positive classes
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    logging.info("Model evaluation completed.")

if __name__ == "__main__":
    car_data = load_car_evaluation_data()
    if car_data is not None:
        X_train, X_test, y_train, y_test = preprocess_data(car_data.copy())
        model = load_model()
        evaluate_model(model, X_test, y_test)
    else:
        logging.error("Model evaluation failed due to data loading error.")
