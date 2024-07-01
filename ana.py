import pandas as pd
import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('Reviews.csv',delimiter=';')

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment and subjectivity
sentiment_scores = []
blob_subj = []
review_text = df['Text']  # Assuming the reviews are in a column named 'Text'

for review in review_text:
    sentiment_scores.append(analyzer.polarity_scores(review)['compound'])
    blob = TextBlob(review)
    blob_subj.append(blob.subjectivity)

# Classify sentiment based on the Vader scores
sentiment_classes = []
for score in sentiment_scores:
    if score > 0.8:
        sentiment_classes.append("Highly positive")
    elif score > 0.4:
        sentiment_classes.append("Positive")
    elif -0.4 <= score <= 0.4:
        sentiment_classes.append("Neutral")
    elif score < -0.4:
        sentiment_classes.append("Negative")
    else:
        sentiment_classes.append("Highly Negative")

# Streamlit app
st.title("Sentiment Analysis on Customer Feedback")

# User input for feedback
user_input = st.text_area("Enter the feedback")
if user_input:
    blob = TextBlob(user_input)
    user_sentiment_score = analyzer.polarity_scores(user_input)['compound']

    if user_sentiment_score > 0.8:
        user_sentiment_class = "Highly positive"
    elif user_sentiment_score > 0.4:
        user_sentiment_class = "Positive"
    elif -0.4 <= user_sentiment_score <= 0.4:
        user_sentiment_class = "Neutral"
    elif user_sentiment_score < -0.4:
        user_sentiment_class = "Negative"
    else:
        user_sentiment_class = "Highly Negative"

    st.write("**VADER Sentiment Class:**", user_sentiment_class, "**VADER Sentiment Score:**", user_sentiment_score)
    st.write("**TextBlob Polarity:**", blob.sentiment.polarity, "**TextBlob Subjectivity:**", blob.sentiment.subjectivity)

# Clean text input
pre = st.text_input('Clean Text')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))
else:
    st.write("No text has been provided for cleaning.")

# Graphical representation of data
st.subheader("Graphical Representation of Data")
plt.figure(figsize=(10, 6))

sentiment_scores_by_class = {k: [] for k in set(sentiment_classes)}
for score, sentiment_class in zip(sentiment_scores, sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(score)

for sentiment_class, scores in sentiment_scores_by_class.items():
    plt.hist(scores, label=sentiment_class, alpha=0.5)

plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.title("Score Distribution by Class")
plt.legend()
st.pyplot(plt)

# Data frame with sentiment analysis results
df['Sentiment Class'] = sentiment_classes
df['Sentiment Score'] = sentiment_scores
df['Subjectivity'] = blob_subj

new_df = df[['Score', 'Text', 'Sentiment Score', 'Sentiment Class', 'Subjectivity']]
st.subheader("Input DataFrame")
st.dataframe(new_df.head(50), use_container_width=True)
