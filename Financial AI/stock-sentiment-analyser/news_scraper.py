import requests

def get_news_headlines(stock_name, api_key):
    url = f"https://newsapi.org/v2/everything?q=stock&{api_key}"
    res = requests.get(url).json()
    print([article['title'] for article in res['articles'][:10]])
    return [article['title'] for article in res['articles'][:10]]
