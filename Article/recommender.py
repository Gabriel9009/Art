import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "articles.csv")
    return pd.read_csv(csv_path)

df = load_data()
articles = df["Article"].tolist()

# Vectorize articles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(articles)

# Similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_articles(title, top_n=5):
    if title not in df['Title'].values:  # Make sure column name matches your CSV
        return []
    
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    return [{
        "Title": df.iloc[i[0]]['Title'],
        "Sponsored": df.iloc[i[0]].get('Sponsored', False)  # Safely get Sponsored flag
    } for i in sim_scores]
