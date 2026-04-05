import requests
from transformers import pipeline
from config.settings import NEWS_API_KEY
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pulsede.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fetch_headlines():
    try:
        logger.info("Fetching headlines from NewsAPI...")
        url = f"https://newsapi.org/v2/everything?q=stock+market&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        logger.info(f"Fetched {len(articles)} articles successfully")
        return [article["title"] for article in articles[:5]]
    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timed out")
        return []
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching headlines: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching headlines: {e}")
        return []

def analyze_sentiment(headlines):
    if not headlines:
        logger.warning("No headlines to analyze, skipping sentiment analysis")
        return []
    try:
        logger.info("Running FinBERT sentiment analysis...")
        sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
        results = sentiment_pipeline(headlines)
        for headline, result in zip(headlines, results):
            logger.info(f"Sentiment: {result['label']} ({round(result['score'] * 100, 2)}%) - {headline[:60]}...")
        return results
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return []

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