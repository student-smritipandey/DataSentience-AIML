from scripts.news_scraper import fetch_crypto_news
from models.sentiment_model import CryptoSentimentAnalyzer

def main():
    print("📰 Crypto News Sentiment Analyzer\n")
    api_key = "2a28b5a6a3e740cbbd12fa10442e29dd"
    if not api_key:
        print("❌ API key is required.")
        return

    crypto = input("Enter crypto topic (e.g., bitcoin, ethereum): ").lower().strip() or "crypto"
    num_articles = int(input("How many news articles to analyze? (default: 10): ") or 10)

    print(f"\nFetching {num_articles} articles about {crypto}...\n")
    news_list = fetch_crypto_news(api_key, query=crypto, page_size=num_articles)

    if not news_list:
        print("⚠️ No articles found.")
        return

    analyzer = CryptoSentimentAnalyzer()
    summary = {"positive": 0, "neutral": 0, "negative": 0}

    for idx, news in enumerate(news_list, 1):
        sentiment = analyzer.predict_sentiment(news)
        label = max(sentiment, key=sentiment.get)
        summary[label] += 1
        print(f"[{idx}] {label.upper()} — {news[:100]}...\n")

    print("🔍 Summary of Sentiment Analysis:")
    for sentiment, count in summary.items():
        print(f"{sentiment.capitalize()}: {count} articles")

if __name__ == "__main__":
    main()
