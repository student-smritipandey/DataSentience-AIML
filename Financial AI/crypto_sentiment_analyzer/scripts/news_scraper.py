import requests

def fetch_crypto_news(api_key, query="crypto", page_size=10):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    return [article["title"] + ". " + article.get("description", "") for article in data.get("articles", [])]
