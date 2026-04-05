import requests
from transformers import pipeline
from config.settings import NEWS_API_KEY

def fetch_headlines():
    url = f"https://newsapi.org/v2/everything?q=stock+market&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [article["title"] for article in articles[:5]]

def analyze_sentiment(headlines):
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    results = sentiment_pipeline(headlines)
    for headline, result in zip(headlines, results):
        print(f"\nHeadline: {headline}")
        print(f"Sentiment: {result['label']} (confidence: {round(result['score'] * 100, 2)}%)")
    return results

def save_results(headlines, results):
    import json
    from datetime import datetime
    
    output = []
    for headline, result in zip(headlines, results):
        output.append({
            "timestamp": datetime.now().isoformat(),
            "headline": headline,
            "sentiment": result["label"],
            "confidence": round(result["score"] * 100, 2)
        })
    
    with open("data/results.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print("\nResults saved to data/results.json")

if __name__ == "__main__":
    print("Fetching headlines...")
    headlines = fetch_headlines()
    print("Running FinBERT sentiment analysis...\n")
    results = analyze_sentiment(headlines)
    save_results(headlines, results)