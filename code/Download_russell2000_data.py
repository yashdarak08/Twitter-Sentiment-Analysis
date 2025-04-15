import yfinance as yf
import pandas as pd

# Define the ticker symbol for Russell 2000 (^RUT on Yahoo Finance)
ticker = '^RUT'

# Define the start and end dates
start_date = '2024-01-01'
end_date = '2024-10-15'

# Download the Russell 2000 historical data from Yahoo Finance
russell_2000_data = yf.download(ticker, start=start_date, end=end_date)

# Keep only the 'Adj Close' and the index, which is the 'Date'
russell_2000_data = russell_2000_data[['Adj Close']]

# Reset the index so 'Date' becomes a column
russell_2000_data.reset_index(inplace=True)

# Display the first few rows of the downloaded data
print(russell_2000_data.head())

# Save the data to a CSV file with only Date and Adj Close
russell_2000_data.to_csv('russell_2000_adj_close_2024.csv', index=False)
