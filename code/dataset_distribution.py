import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./dataset/tweets_with_contextual_sentiment.csv')

# Count the occurrences of each classification in the last column
classification_counts = df['Sentiment_Classification'].value_counts()

# Create a bar plot for the distribution of sentiment classification
plt.figure(figsize=(10, 6))
classification_counts.plot(kind='bar', color='skyblue')

# Add title and labels
plt.title('Distribution of Sentiment Classification', fontsize=16)
plt.xlabel('Sentiment Classification', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Save the bar plot as an image
plt.savefig('sentiment_classification_distribution.png')

# Show the plot
plt.tight_layout()
plt.show()
