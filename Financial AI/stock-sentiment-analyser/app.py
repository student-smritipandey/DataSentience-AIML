import streamlit as st
from news_scraper import get_news_headlines
from sentiment_model import analyze_sentiment
from utils import predict_market_impact
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model & tokenizer once at the top
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="📊 Stock Sentiment Analyzer", layout="centered")
st.title("📊 Stock Sentiment Analyzer")
st.write("Analyze news sentiment for your favorite stock and predict market mood.")

stock_input = st.text_input("Enter stock/company name (e.g., TCS, INFY, Tata Consultancy Services)")

# Mapping for better NewsAPI results
keyword_map = {
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "RELIANCE": "Reliance Industries"
}

api_key = "your_news_api_key_here"  # Replace with your actual NewsAPI key

if stock_input:
    query = keyword_map.get(stock_input.upper(), stock_input)
    st.info(f"Fetching news for: **{query}**")

    headlines = get_news_headlines(query, api_key)

    if not headlines:
        st.error("❌ No relevant news found. Try a broader or correct term.")
    else:
        st.subheader("📰 Top Headlines")
        for i, headline in enumerate(headlines, 1):
            st.markdown(f"{i}. {headline}")

        st.subheader("🧠 Sentiment Analysis")
        sentiment_scores = analyze_sentiment(headlines, tokenizer, model)

        st.write("### 📊 Sentiment Breakdown")
        st.bar_chart(sentiment_scores)

        impact = predict_market_impact(sentiment_scores)
        st.write(f"### 🔮 Market Prediction: **{impact}**")
