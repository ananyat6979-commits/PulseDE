import requests
from config.settings import NEWS_API_KEY

def fetch_headlines():
    url = f"https://newsapi.org/v2/everything?q=stock+market&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    for article in articles[:5]:
        print(article["title"])

if __name__ == "__main__":
    fetch_headlines()