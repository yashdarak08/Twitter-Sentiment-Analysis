import pandas as pd
import matplotlib.pyplot as plt


# Load both datasets
df_with_dates = pd.read_csv('./dataset/dataset1_preprocessed_tweets_with_dates.csv')
df_without_dates = pd.read_csv('./dataset/dataset2_preprocessed_tweets_with_dates.csv')

# Rename the 'Text' column in the first dataframe to 'Tweet' for consistency
df_with_dates.rename(columns={'Text': 'Tweet'}, inplace=True)

# Convert the 'Date' column to mm/dd/yyyy in the first dataframe
df_with_dates['Date'] = pd.to_datetime(df_with_dates['Date']).dt.strftime('%m/%d/%Y')

# In the second dataframe, add the year 2024 to the 'Date' column and convert to mm/dd/yyyy format
df_without_dates['Date'] = pd.to_datetime(df_without_dates['Date'] + ' 2024', format='%b-%d %Y').dt.strftime('%m/%d/%Y')

# Concatenate both dataframes, keeping only 'Date' and 'Tweet' columns
combined_df = pd.concat([df_with_dates[['Date', 'Tweet']], df_without_dates[['Date', 'Tweet']]], ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('combined_tweets.csv', index=False)


