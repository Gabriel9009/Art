import streamlit as st
from recommender import recommend_articles

st.set_page_config(page_title="Smart Article Recommender")

st.title("📚 Smart Article Recommender")
st.write("Paste an article title to get recommendations:")

# Show available titles for reference
try:
    from recommender import df
    st.caption(f"Available articles: {', '.join(df['Title'].unique()[:3])}...")
except:
    pass

article_title = st.text_input("Enter article title:")

if st.button("Recommend"):
    if article_title:
        with st.spinner("Finding great reads..."):
            recs = recommend_articles(article_title)
        
        if not recs:
            st.warning("No recommendations found. Try a different title.")
        else:
            st.success("Here are some recommendations:")
            for i, rec in enumerate(recs):
                tag = " 🌟 *Sponsored*" if rec.get("Sponsored", False) else ""
                st.markdown(f"**{i+1}.** {rec['Title']}{tag}")
    else:
        st.warning("Please enter an article title.")
