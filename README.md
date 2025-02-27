# Safety-Critical-Automotive-CAN-Anomaly-Detection

## Abstract

This project presents a study on forecasting hourly public transport demand using the Prophet model. The study utilizes a dataset of hourly passenger counts for various transport modes in an urban environment. Feature engineering techniques are employed to enhance the dataset, and the Prophet model is trained to predict future demand. A comprehensive evaluation is conducted, including a range of metrics, visualizations, and statistical tests, to assess the model's performance and identify areas for potential improvement.

## 1. Introduction

Accurate forecasting of public transport demand is crucial for efficient resource allocation, route optimization, and service planning. This study focuses on developing a predictive model using the Prophet model, a powerful time series forecasting tool developed by Facebook. The model's ability to handle seasonality and trend makes it well-suited for this task.

## 2. Data and Methodology

### 2.1 Data Acquisition

The study utilizes the "Hourly Public Transport Demand" dataset sourced from Kaggle. This dataset comprises hourly passenger counts for different transport modes (bus, tram, subway) within an urban area.

### 2.2 Feature Engineering

To enhance the dataset and improve the model's predictive capabilities, several feature engineering techniques are employed:

* **Time-based features:** Hour of the day, day of the week, and month are extracted from the datetime index to capture temporal patterns.
* **Lag features:** Previous hourly and daily passenger counts are included to capture temporal dependencies.
* **Scaling:** Numerical features are standardized to ensure consistent scaling across variables.

### 2.3 Model Training

The Prophet model is trained on the preprocessed dataset. Prophet is an additive regression model that incorporates seasonality and trend components. The model automatically detects changes in trends and incorporates seasonality using Fourier series and dummy variables.

## 3. Evaluation

A comprehensive evaluation of the model's performance is conducted using various metrics, visualizations, and statistical tests:

### 3.1 Metrics

The following metrics are calculated to assess the accuracy of the predictions:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R-squared (R2)
* Mean Absolute Percentage Error (MAPE)
* Explained Variance Score
* Median Absolute Error (MedAE)
* Mean Squared Log Error (MSLE)

### 3.2 Visualizations

Visualizations are employed to gain insights into the model's behavior and the accuracy of the predictions:

* Actual vs. Predicted values
* Residuals plot
* Histogram of residuals
* ACF and PACF plots of residuals
* Q-Q plot of residuals

### 3.3 Statistical Tests

Statistical tests are conducted to analyze the residuals and assess the model's assumptions:

* Shapiro-Wilk test for normality of residuals
* Box-Cox transformation to address non-normality
* Augmented Dickey-Fuller (ADF) test for stationarity
* Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity

## 4. Results and Discussion

The results of the evaluation are presented and discussed, highlighting the model's strengths and weaknesses. The analysis of the residuals and the statistical tests provide insights into potential areas for model improvement.

## 5. Conclusion

This study demonstrates the effectiveness of the Prophet model in forecasting public transport demand. The feature engineering techniques and the comprehensive evaluation framework contribute to a robust and reliable prediction model. Future work may involve exploring additional features, experimenting with other forecasting models, and deploying the model for real-time predictions.

## 6. References

* Prophet: https://facebook.github.io/prophet/
* Hourly Public Transport Demand Dataset: [[Kaggle Dataset Link]](https://www.kaggle.com/datasets/serdargundogdu/municipality-bus-utilization/code)

## 7. Code and Reproducibility

The code for this project is available on GitHub: [GitHub Repository Link]

The repository includes detailed instructions on how to reproduce the results and conduct further experiments.
