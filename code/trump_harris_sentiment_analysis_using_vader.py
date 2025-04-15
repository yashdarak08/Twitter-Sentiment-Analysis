import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")



# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the dataset
file_path = './dataset/combined_tweets_with_sentiment.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# def lemmatize_text(text):
#     doc = nlp(text)
#     return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# df['Tweet_Lemmatized'] = df['Tweet'].apply(lemmatize_text)

# Define expanded keyword lists for Trump and Harris
trump_keywords = ['trump', 'donald', 'president trump', 'djt', 'don', 'former president', '45', 
                  'gop', 'republicans', 'republican party', '#maga', '#trump2024', 'make america great again']
harris_keywords = ['harris', 'kamala', 'kamalaharris', 'vp harris', 'vice president harris', 'kamalah', 
                   'democrats', 'democratic party', '#bidenharris', '#harris2024']

# Function to get the sentiment of a specific window of text around a keyword
def get_contextual_sentiment(text, keyword):
    if isinstance(text, str):  # Check if the text is a string
        # Parse the text using spaCy
        doc = nlp(text)

        # Find the location of the keyword in the text
        for token in doc:
            if keyword.lower() in token.text.lower():
                # Get a window of words around the keyword (5 words before and after)
                start = max(0, token.i - 5)
                end = min(len(doc), token.i + 6)
                context = doc[start:end].text

                # Analyze the sentiment of the context using VADER
                sentiment_score = analyzer.polarity_scores(context)['compound']
                return sentiment_score
    return 0.0

# Function to analyze the sentiment towards Trump and Harris based on context
def analyze_sentiment_contextual(row):
    tweet = row['Tweet']
    trump_sentiment = 0.0
    harris_sentiment = 0.0
    
    # If Trump is mentioned, get the contextual sentiment around his name
    if isinstance(tweet, str):  # Check if tweet is a string
        for keyword in trump_keywords:
            if keyword.lower() in tweet.lower():
                trump_sentiment = get_contextual_sentiment(tweet, keyword)
                break  # Stop after the first Trump keyword is found

        # If Harris is mentioned, get the contextual sentiment around her name
        for keyword in harris_keywords:
            if keyword.lower() in tweet.lower():
                harris_sentiment = get_contextual_sentiment(tweet, keyword)
                break  # Stop after the first Harris keyword is found

    return trump_sentiment, harris_sentiment

# Apply the function to the dataset
df[['Trump_Context_Sentiment', 'Harris_Context_Sentiment']] = df.apply(analyze_sentiment_contextual, axis=1, result_type='expand')

# Function to classify sentiment based on the contextual sentiment scores
def classify_sentiment(row):
    trump_sentiment = row['Trump_Context_Sentiment']
    harris_sentiment = row['Harris_Context_Sentiment']
    
    # Set a small threshold for neutral scores (close to 0)
    neutral_threshold = 0.05
    
    # If both Trump and Harris are mentioned
    if pd.notnull(trump_sentiment) and pd.notnull(harris_sentiment):
        # Check if both scores are neutral
        if abs(trump_sentiment) < neutral_threshold and abs(harris_sentiment) < neutral_threshold:
            return 'Neutral'
        
        # Check for mixed sentiment
        if trump_sentiment > harris_sentiment:
            return 'Pro-Trump'
        elif harris_sentiment > trump_sentiment:
            return 'Pro-Harris'
        elif trump_sentiment > 0 and harris_sentiment < 0:
            return 'Pro-Trump / Anti-Harris'
        elif harris_sentiment > 0 and trump_sentiment < 0:
            return 'Pro-Harris / Anti-Trump'
        else:
            return 'Mixed'
    
    # If only Trump is mentioned
    if pd.notnull(trump_sentiment):
        if abs(trump_sentiment) < neutral_threshold:
            return 'Neutral'
        return 'Pro-Trump' if trump_sentiment > 0 else 'Anti-Trump'
    
    # If only Harris is mentioned
    if pd.notnull(harris_sentiment):
        if abs(harris_sentiment) < neutral_threshold:
            return 'Neutral'
        return 'Pro-Harris' if harris_sentiment > 0 else 'Anti-Harris'
    
    # Default to Neutral if nothing else applies
    return 'Neutral'


# Apply the classification function
df['Sentiment_Classification'] = df.apply(classify_sentiment, axis=1)

# Save the updated dataframe to a new CSV file
df.to_csv('./dataset/tweets_with_contextual_sentiment.csv', index=False)

# Display the first few rows of the updated dataframe
#import ace_tools as tools; tools.display_dataframe_to_user(name="Contextual Sentiment Classification", dataframe=df)

# Load the dataset
file_path = './dataset/tweets_with_contextual_sentiment.csv'  # Replace with the path to your dataset
df = pd.read_csv(file_path)

# Plotting the distribution of sentiment classifications
plt.figure(figsize=(10, 6))
df['Sentiment_Classification'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Sentiment Classifications', fontsize=16)
plt.xlabel('Sentiment Classification', fontsize=14)
plt.ylabel('Number of Tweets', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('./Result/sentiment_classification_distribution.png')

# Show the plot
plt.show()

