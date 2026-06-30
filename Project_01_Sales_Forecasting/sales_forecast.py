import pandas as pd
import numpy as np
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

if __name__ == "__main__":
    main()
