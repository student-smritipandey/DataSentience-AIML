import streamlit as st
from scripts.news_scraper import fetch_crypto_news
from models.sentiment_model import CryptoSentimentAnalyzer

def main():
    st.title("📈 Crypto News Sentiment Analyzer")
    st.markdown("Analyze real-time crypto news articles for **Positive**, **Neutral**, or **Negative** sentiment.")

    # --- Input: API Key ---
    api_key = st.text_input("🔐 Enter your NewsAPI key", type="password")
    if not api_key:
        st.warning("Please enter your NewsAPI key to proceed.")
        st.stop()

    # --- Input: Topic and Number of Articles ---
    crypto = st.text_input("🪙 Enter crypto topic", value="bitcoin")
    num_articles = st.slider("📄 Number of news articles to analyze", min_value=5, max_value=50, value=10)

    if st.button("🔍 Analyze Sentiment"):
        with st.spinner("Fetching and analyzing news..."):
            news_list = fetch_crypto_news(api_key, query=crypto, page_size=num_articles)

            if not news_list:
                st.error("No articles found.")
                return

            analyzer = CryptoSentimentAnalyzer()
            summary = {"positive": 0, "neutral": 0, "negative": 0}

            for idx, news in enumerate(news_list, 1):
                sentiment = analyzer.predict_sentiment(news)
                label = max(sentiment, key=sentiment.get)
                summary[label] += 1

                # Display each article with sentiment
                st.markdown(f"**{idx}. [{label.upper()}]** — {news[:150]}...")

            # --- Summary Chart ---
            st.subheader("📊 Sentiment Summary")
            st.write(summary)

if __name__ == "__main__":
    main()
