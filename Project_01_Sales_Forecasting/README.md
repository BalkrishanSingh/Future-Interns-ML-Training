# Superstore Sales Forecasting

This repository contains my submission for the sales forecasting assessment. The goal of this project is to build a basic machine learning pipeline to forecast future sales using historical data from the Superstore dataset.

## Files Included
- `Sample - Superstore.csv`: The raw dataset.
- `Sales_Forecasting_Colab.ipynb`: A Jupyter Notebook intended to be run in Google Colab. It does data preparation, model training, evaluation, and plotting.
- `sales_forecast.py`: A standalone Python script for the same.

## Methodology
I chose a straightforward Random Forest Regressor approach, using time-based features (`Time_Index` to capture general trend and `Month` to capture seasonality). I went with this method because it handles non-linear patterns (like monthly seasonality) much better than simple linear models, while still remaining highly interpretable and easy to explain to stakeholders.

The pipeline performs the following steps:
1. Loads the CSV and parses dates.
2. Aggregates the daily sales data into monthly totals to reduce noise.
3. Splits the data, holding out the last 6 months for testing.
4. Fits a Random Forest Regressor on the training set.
5. Evaluates the model using MAE and RMSE.
6. Forecasts sales for the next 12 months and visualizes the results (including test data predictions to prove accuracy) using Matplotlib.
