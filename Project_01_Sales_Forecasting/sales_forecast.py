import pandas as pd
import numpy as np
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
    print(f"Monthly sales record count: {len(monthly_sales)}")
    print(monthly_sales.head())

if __name__ == "__main__":
    main()
