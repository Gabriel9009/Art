import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

# Get the absolute path to articles.csv in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "articles.csv")

# Verify the file exists
if not os.path.exists(csv_path):
    st.error(f"Error: Cannot find articles.csv at:\n{csv_path}")
    st.error(f"Files in this directory: {os.listdir(current_dir)}")
    st.stop()

# Load the data
try:
    df = pd.read_csv(csv_path)
    articles = df["Article"].tolist()
    
    # Vectorize articles
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(articles)
    
    # Similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.stop()

def recommend_articles(title, top_n=5):
    if title not in df['title'].values:
        return []
    
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    return [df.iloc[i[0]]['title'] for i in sim_scores]

# Simple UI
st.title("Article Recommender")
selected = st.selectbox("Choose an article:", df['title'].unique())
recommendations = recommend_articles(selected)

if recommendations:
    st.subheader("Recommended Articles:")
    for i, title in enumerate(recommendations, 1):
        st.write(f"{i}. {title}")
else:
    st.warning("No recommendations found")
