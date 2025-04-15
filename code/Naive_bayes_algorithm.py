import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# Load the dataset (tweets and sentiment classifications)
df = pd.read_csv('dataset/tweets_with_contextual_sentiment.csv')

# Handle missing values in the 'Tweet' column
df['Tweet'] = df['Tweet'].fillna('')  # Replace NaN with empty string

# Load the top features from the Chi-Square test results
chi2_df = pd.read_csv('./Result/chi_square_feature_scores.csv')

# Get the top 100 features based on Chi-Square score (adjust this number if necessary)
top_features = chi2_df['Feature'].head(700).tolist()

# # Re-vectorize using only the top selected features
# vectorizer_top = CountVectorizer(stop_words='english', vocabulary=top_features)
# X_top = vectorizer_top.fit_transform(df['Tweet'])  # Use the 'Tweet' column for feature extraction

# # Target labels (sentiment classification)
# y = df['Sentiment_Classification']

# Use TF-IDF Vectorizer instead of CountVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(df['Tweet'])
y = df['Sentiment_Classification']

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier using the selected features
nb_classifier = MultinomialNB(alpha=0.01)
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Naive Bayes Classifier Accuracy: {accuracy * 100:.2f}%')

# Perform cross-validation
cross_val_scores = cross_val_score(nb_classifier, X_train, y_train, cv=10)
print(f'Cross-Validation Accuracy: {cross_val_scores.mean() * 100:.2f}%')

# Optional: Save the predictions to a CSV file for further analysis
predictions_df = pd.DataFrame({'Tweet': df['Tweet'].iloc[y_test.index], 'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('./Result/naive_bayes_predictions.csv', index=False)

print("Predictions saved to 'naive_bayes_predictions.csv'")
