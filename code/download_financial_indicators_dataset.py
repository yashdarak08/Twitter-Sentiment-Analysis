import yfinance as yf

# Define the start and end dates
start_date = '2024-01-01'
end_date = '2024-10-15'

# Download S&P 500 data and keep only Date and Adj Close columns
sp500 = yf.download('^GSPC', start=start_date, end=end_date)[['Adj Close']]
sp500.reset_index(inplace=True)  # Reset index to keep Date as a column

# Download Gold data (COMEX Gold Futures) and keep only Date and Adj Close columns
gold = yf.download('GC=F', start=start_date, end=end_date)[['Adj Close']]
gold.reset_index(inplace=True)  # Reset index to keep Date as a column

# Download 10-year Treasury yield and keep only Date and Adj Close columns
treasury_yield = yf.download('^TNX', start=start_date, end=end_date)[['Adj Close']]
treasury_yield.reset_index(inplace=True)  # Reset index to keep Date as a column

# Save datasets to CSV files with Date and Adj Close columns
sp500.to_csv('sp500_2024_adj_close.csv', index=False)
gold.to_csv('gold_2024_adj_close.csv', index=False)
treasury_yield.to_csv('treasury_yield_2024_adj_close.csv', index=False)

# Print confirmation
print("Datasets with Date and Adj Close saved as CSV files.")
