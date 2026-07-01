import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_csv('Sample - Superstore.csv', encoding='latin-1')
    
    print("Preparing data...")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df[['Order Date', 'Sales']]
    
    # Aggregate to monthly sales
    monthly_sales = df.resample('ME', on='Order Date').sum().reset_index()
    
    # Feature engineering
    monthly_sales['Time_Index'] = np.arange(len(monthly_sales))
    monthly_sales['Month'] = monthly_sales['Order Date'].dt.month
    
    # Train/test split (hold out last 6 months for evaluation)
    test_size = 6
    train_size = len(monthly_sales) - test_size
    
    X = monthly_sales[['Time_Index', 'Month']]
    y = monthly_sales['Sales']
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n--- Evaluation on Test Set ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    print("\nGenerating 12-month forecast...")
    last_date = monthly_sales['Order Date'].max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=12, freq='ME')
    
    future_X = pd.DataFrame({
        'Time_Index': np.arange(len(monthly_sales), len(monthly_sales) + 12),
        'Month': future_dates.month
    })
    
    future_predictions = model.predict(future_X)
    future_sales_df = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': future_predictions})
    
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    
    # Plot entire historical data
    plt.plot(monthly_sales['Order Date'], monthly_sales['Sales'], label='Historical Data', color='blue', marker='.')
    
    # Plot the predictions made on the test data (last 6 months)
    test_dates = monthly_sales['Order Date'].iloc[train_size:]
    plt.plot(test_dates, y_pred, label='Test Data Predictions', color='orange', linestyle='--', marker='.')
    
    # Plot the future forecast
    plt.plot(future_sales_df['Date'], future_sales_df['Predicted_Sales'], label='Future Forecast', color='red', linestyle='--', marker='.')
    
    plt.title('Superstore Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=last_date, color='black', linestyle=':', label='Forecast Start')
    
    plt.show()

if __name__ == "__main__":
    main()
