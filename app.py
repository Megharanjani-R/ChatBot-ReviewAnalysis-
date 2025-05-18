from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load models
model = pickle.load(open('model.pkl', 'rb'))
clf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
search_vectorizer = pickle.load(open('search_vectorizer.pkl', 'rb'))

# Load dataset
df = pd.read_csv("clean_reviews.csv")
df = df[['content', 'score']].dropna()
review_embeddings = search_vectorizer.transform(df['content'])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', messages=[])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['review']
    bot_response = analyze_and_respond(user_message)

    messages = [
        {"sender": "user", "text": user_message},
        {"sender": "bot", "text": bot_response}
    ]
    return render_template('index.html', messages=messages)

def analyze_and_respond(user_message):
    # Predict sentiment
    tfidf = clf_vectorizer.transform([user_message])
    sentiment = model.predict(tfidf)[0]
    sentiment_text = "positive" if sentiment == 1 else "negative"

    # Search for similar reviews
    query_vec = search_vectorizer.transform([user_message])
    sims = cosine_similarity(query_vec, review_embeddings).flatten()
    top_indices = sims.argsort()[-3:][::-1]
    relevant_reviews = df.iloc[top_indices]

    # Compose response
    summary = "Here's what others are saying:\n"
    for _, row in relevant_reviews.iterrows():
        snippet = row['content'][:120].strip().replace("\n", " ")
        summary += f"- (Score {row['score']}): \"{snippet}...\"\n"

    summary += f"\nYou seem to feel {sentiment_text} about this topic."
    return summary

if __name__ == "__main__":
    app.run(debug=True)
