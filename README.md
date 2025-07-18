# Stock Price Predictor
This project is a stock price prediction application that leverages an XGBoost machine learning model to estimate the closing price of a stock based on key fundamental and technical financial indicators. It features a user-friendly web interface built with Streamlit for easy interaction.

In addition to the prediction application, this repository includes a comprehensive Jupyter Notebook that details the entire data science workflow, from initial data exploration and cleaning to feature engineering and model tuning.

## Features
Stock Price Prediction: Estimates the closing price of a stock using an XGBoost model.

Interactive Web App: A simple and intuitive user interface created with Streamlit.

Fundamental and Technical Indicators: Utilizes a combination of essential financial metrics for prediction:

Total Revenue (in billions)

Earnings Per Share (EPS)

Price-to-Earnings (P/E) Ratio

GICS Sector

Technical indicators (placeholders in the app, but part of the model)

In-depth Analysis: The accompanying Jupyter Notebook provides a thorough walkthrough of the data analysis, feature engineering, and model comparison process.

## Data Science Workflow (stock_pred_xgboost.ipynb)
The Jupyter Notebook documents the following key stages of the project:

Data Loading and Initial Exploration: Importing the necessary datasets (prices-split-adjusted.csv, fundamentals.csv, securities.csv) and performing an initial assessment of their structure and content.

Data Cleaning and Preparation: Handling missing values, standardizing column names, and identifying and addressing outliers to ensure data quality.

Exploratory Data Analysis (EDA): Conducting univariate, bivariate, and multivariate analysis to uncover patterns, correlations, and key trends in the data.

Feature Engineering: Creating new features from the existing data, including technical indicators like Moving Averages (MA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).

Model Building and Evaluation: Training and evaluating both a baseline Linear Regression model and a more advanced XGBoost Regressor.

Model Tuning: Using GridSearchCV to find the optimal hyperparameters for the XGBoost model to improve its predictive performance.

## Installation
To run this project locally, please ensure you have Python installed. Then, install the required libraries using pip:

pip install streamlit pandas numpy scikit-learn xgboost pmdarima joblib

## Usage
Clone the repository:

git clone <repository-url>
cd <repository-directory>

## Run the Streamlit application:
Make sure the following files are in the same directory as app.py:

xgb_stock_predictor.joblib

scaler.joblib

model_columns.joblib

Execute the following command in your terminal:

streamlit run app.py

Interact with the app:

Open the provided local URL in your web browser.

Use the sidebar to input the financial indicators for the stock you wish to analyze.

Click the "Predict" button to see the estimated closing price.

Model Information
Prediction App (app.py): The web application uses a pre-trained XGBoost model (xgb_stock_predictor.joblib) for its predictions. This model was trained on a combination of fundamental and technical indicators to provide robust estimations.

Model Development (stock_pred_xgboost.ipynb): The notebook details the development and tuning of the XGBoost model, which was selected for its ability to handle complex, non-linear relationships in financial data.

## Files in this Repository
app.py: The main file for the Streamlit web application.

stock_pred_xgboost.ipynb: A Jupyter Notebook containing the complete data analysis and model development process.

xgb_stock_predictor.joblib: The serialized, pre-trained XGBoost model.

scaler.joblib: The scaler object used for data normalization.

model_columns.joblib: A list of columns used during model training.
