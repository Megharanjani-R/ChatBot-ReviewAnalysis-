import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load and clean data
df = pd.read_csv("clean_reviews.csv")
df = df[['content', 'score']].dropna()
df['sentiment'] = df['score'].apply(lambda x: 1 if x >= 3 else 0)

# TF-IDF vectorizer for sentiment
clf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = clf_vectorizer.fit_transform(df['content'])
y = df['sentiment']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save sentiment model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(clf_vectorizer, open("tfidf.pkl", "wb"))

# Save separate vectorizer for similarity search
search_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
search_vectorizer.fit(df['content'])
pickle.dump(search_vectorizer, open("search_vectorizer.pkl", "wb"))
