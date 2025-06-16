from news_scraper import get_news_headlines
from sentiment_model import analyze_sentiment
from utils import predict_market_impact        
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    print("📊 Stock Sentiment Analyzer (CLI)\n")

    stock_name = input("Enter stock/company name (e.g., TCS, INFY, Tata Consultancy Services): ").strip()

    # Map common tickers → full names for better NewsAPI results
    keyword_map = {
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "RELIANCE": "Reliance Industries"
    }
    query = keyword_map.get(stock_name.upper(), stock_name)

    api_key =#replace with your news api  key

    print("\n🔍 Fetching news headlines…")
    headlines = get_news_headlines(query, api_key)

    if not headlines:
        print("❌ No relevant news found. Try a broader term.")
        return

    print(f"\n📰 Top {len(headlines)} headlines:")
    for i, h in enumerate(headlines, 1):
        print(f"{i}. {h}")

    print("\n🧠 Analyzing sentiment…")

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    sentiment_scores = analyze_sentiment(headlines, tokenizer, model)

    print("\n📊 Sentiment breakdown:")
    for label, pct in sentiment_scores.items():
        print(f"- {label.capitalize()}: {pct}%")

    impact = predict_market_impact(sentiment_scores)
    print("\n🔮 Market impact prediction:", impact)

if __name__ == "__main__":
    main()
