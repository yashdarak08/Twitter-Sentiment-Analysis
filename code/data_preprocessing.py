

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to remove emojis using regex
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Function to preprocess the tweet text
def preprocess_text(text):
    # Remove emojis
    text = remove_emojis(text)
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return ' '.join(tokens)

# Process dataset1
def process_dataset1(file_path):
    df = pd.read_csv(file_path)
    
    # Keep only the 'Date' and 'Text' columns
    df = df[['Date', 'Text']]
    
    # Preprocess the 'Text' column
    df['Text'] = df['Text'].apply(preprocess_text)
    
    # Save the preprocessed data to a new CSV file
    df.to_csv('./dataset/dataset1_preprocessed_tweets_with_dates.csv', index=False)
    
    # Display the first few rows of the preprocessed data
    print(df.head())

# Process dataset2
def process_dataset2(file_path):
    df = pd.read_csv(file_path)
    
    # Keep only the 'Date' and 'Tweet' columns
    df = df[['Date', 'Tweet']]
    
    # Preprocess the 'Tweet' column
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    
    # Save the preprocessed data to a new CSV file
    df.to_csv('./dataset/dataset2_preprocessed_tweets_with_dates.csv', index=False)
    
    # Display the first few rows of the preprocessed data
    print(df.head())

# File paths for the datasets
file_path_dataset1 = './dataset/dataset1_uncleaned_tweets_data_with_dates.csv'
file_path_dataset2 = './dataset/dataset2_uncleaned_tweets_data_with_dates.csv'

# Process both datasets
process_dataset1(file_path_dataset1)
process_dataset2(file_path_dataset2)
