import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv('./dataset/tweets_with_contextual_sentiment.csv')

# Check the structure of the dataset (ensure you have the necessary columns)
print(df.head())

# Step 1: Handle missing values in the 'Tweet' column
df['Tweet'] = df['Tweet'].fillna('')  # Fill NaN values with an empty string

# Step 1: Preprocess the data
# We'll use the 'Tweet' column as features (input) and 'Sentiment_Classification' as labels (target)
X = df['Tweet']  # Feature: tweet text
y = df['Sentiment_Classification']  # Target: sentiment classification (Pro-Trump, Pro-Harris, etc.)

# Step 2: Convert the text data into numerical features using CountVectorizer (Bag of Words)
vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Limiting to 1000 features
X_vectorized = vectorizer.fit_transform(X)

# Step 3: Apply Chi-Square Test to select the best features
chi2_scores, p_values = chi2(X_vectorized, y)

# Create a DataFrame to show feature importance based on chi2 score
feature_names = vectorizer.get_feature_names_out()
chi2_df = pd.DataFrame({'Feature': feature_names, 'Chi2_Score': chi2_scores, 'P_Value': p_values})
chi2_df.sort_values(by='Chi2_Score', ascending=False, inplace=True)

# Save the chi-square result to a CSV file
chi2_df.to_csv('./Result/chi_square_feature_scores.csv', index=False)


# Display the top features based on chi-square score
print("Top features based on Chi-Square test:")
print(chi2_df.head(20))