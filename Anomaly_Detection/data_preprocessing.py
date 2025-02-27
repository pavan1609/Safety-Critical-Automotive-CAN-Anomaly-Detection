import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from data_acquisition import load_car_evaluation_data  # Import data loading function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the DataFrame using LabelEncoder.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    logging.info("Encoding categorical columns...")
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
            logging.info(f"Encoded column: {column}")
    return df

def scale_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numerical columns in the DataFrame using StandardScaler.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with scaled numerical columns.
    """
    logging.info("Scaling numerical columns...")
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    logging.info("Numerical columns scaled.")
    return df

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocesses the data by encoding categorical columns, scaling numerical columns,
    and splitting the data into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    logging.info("Preprocessing data...")
    df = encode_categorical_columns(df)
    df = scale_numerical_columns(df)
    X = df.drop('class', axis=1)  # Features
    y = df['class']  # Target variable (anomaly labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    car_data = load_car_evaluation_data()
    if car_data is not None:
        X_train, X_test, y_train, y_test = preprocess_data(car_data.copy()) #copy the data, so that the original dataframe is not changed.
        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}")
        logging.info(f"y_train shape: {y_train.shape}")
        logging.info(f"y_test shape: {y_test.shape}")
    else:
        logging.error("Data preprocessing failed due to data loading error.")
