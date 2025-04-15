import pandas as pd
import matplotlib.pyplot as plt

# Load the sentiment dataset
sentiment_df = pd.read_csv('./dataset/tweets_with_contextual_sentiment.csv')

# Load Polymarket dataset
polymarket_df = pd.read_csv('./dataset/polymarket_daily_election.csv')

# Convert the 'Date' columns to datetime format
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
polymarket_df['Date'] = pd.to_datetime(polymarket_df['Date (UTC)'])

# Merge sentiment and Polymarket dataset on Date column
merged_polymarket = pd.merge(sentiment_df, polymarket_df[['Date', 'Donald Trump', 'Kamala Harris']], on='Date', how='inner')

# Aggregate by Date: take the mean of Trump/Harris sentiment and Polymarket prices for each date
aggregated_polymarket = merged_polymarket.groupby('Date').agg({
    'Trump_Context_Sentiment': 'mean',
    'Harris_Context_Sentiment': 'mean',
    'Donald Trump': 'mean',
    'Kamala Harris': 'mean'
}).reset_index()

# Plot sentiment scores and Polymarket prices
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot Trump and Harris sentiment on the first y-axis (left)
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Score', color='blue')
ax1.plot(aggregated_polymarket['Date'], aggregated_polymarket['Trump_Context_Sentiment'], label='Trump Sentiment (Mean)', color='blue', linestyle='dashed')
ax1.plot(aggregated_polymarket['Date'], aggregated_polymarket['Harris_Context_Sentiment'], label='Harris Sentiment (Mean)', color='pink', linestyle='dotted')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

# Create a second y-axis for Polymarket values (right)
ax2 = ax1.twinx()
ax2.set_ylabel('Polymarket Price', color='purple')
ax2.plot(aggregated_polymarket['Date'], aggregated_polymarket['Donald Trump'], label='Trump Polymarket Price (Mean)', color='blue')
ax2.plot(aggregated_polymarket['Date'], aggregated_polymarket['Kamala Harris'], label='Harris Polymarket Price (Mean)', color='pink')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.legend(loc='upper right')

# Add a title and format the x-axis
plt.title('Trump and Harris Sentiment vs Polymarket Prices (Mean Aggregated) Over Time', fontsize=16)
plt.xticks(rotation=45)
plt.grid(True)

# Save the plot as an image
plt.savefig('./Result/aggregated_sentiment_vs_polymarket_prices.png')

# Show the plot
plt.tight_layout()
plt.show()