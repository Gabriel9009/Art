# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load articles
df = pd.read_csv("articles.csv")
articles = df["Article"].tolist()

# Vectorize articles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(articles)

# Similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend function
def recommend_articles(title, top_n=5):
    if title not in df['title'].values:
        return []

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    recs = []
    for i in sim_scores:
        row = df.iloc[i[0]]
        recs.append({
            "title": row['title'],
            "sponsored": row.get('sponsored', False)
        })
    return recs
