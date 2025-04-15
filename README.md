# Twitter Sentiment Analysis for Presidential Election 2024

In the lead-up to the 2024 U.S. Presidential Election, social media platforms like Twitter play a significant role in shaping public opinion and political discourse. 

This project focuses on Twitter sentiment analysis to evaluate the public's views on key candidates, specifically Donald Trump and Kamala Harris. 

By leveraging machine learning models and sentiment analysis tools, the project classifies tweets as positive, negative, or neutral, and further identifies whether a tweet is biased towards Trump, Harris, or neutral. 

Additionally, the project explores how these sentiments correlate with financial markets and prediction platforms, such as Polymarket. 

## Dataset
 Twitter Data:

dataset1_uncleaned_tweets_data_with_dates.csv: Contains raw, unprocessed tweets with associated dates. This dataset includes the uncleaned text of the tweets and is used as the basis for sentiment analysis.

dataset2_uncleaned_tweets_data_with_dates.csv: Similar to the first dataset, this file contains uncleaned tweet text but for a different set of data.

dataset1_preprocessed_tweets_with_dates.csv: The preprocessed version of the first dataset, where emojis, stopwords, and unnecessary characters have been removed, ready for sentiment analysis.

dataset2_preprocessed_tweets_with_dates.csv: The cleaned version of the second unprocessed tweet dataset, prepared for further analysis.

combined_tweets_with_sentiment.csv: This dataset combines the preprocessed tweets with their sentiment scores (positive, negative, or neutral) and identifies if the tweet is biased towards Trump, Harris, or neutral.

tweets_with_contextual_sentiment.csv: This dataset further analyzes each tweet by including the contextual sentiment score for Trump and Harris individually, allowing for a deeper understanding of bias in each tweet.

Financial Market Data:

gold_2024_adj_close.csv: Contains the daily adjusted close prices for gold from January 2024 to October 2024, used to analyze gold's performance in relation to the election sentiment.

sp500_2024_adj_close.csv: This file tracks the adjusted close prices of the S&P 500 index over the same period, providing insights into how the stock market reacts to public sentiment regarding the election.

russell_2000_adj_close_2024.csv: Records the adjusted close prices of the Russell 2000 index, which focuses on smaller U.S. companies, offering a different perspective on the election's market impact.

treasury_yield_2024_adj_close.csv: Contains data for the U.S. 10-year Treasury Yield, often used as a benchmark for investor sentiment towards government debt and long-term economic outlook.

polymarket_daily_election.csv: Tracks the prediction prices for the 2024 U.S. Presidential Election on Polymarket, a decentralized prediction platform. This dataset is used to correlate market prediction sentiment with public opinion on social media.

## 

