from kafka import KafkaProducer
import json
import requests
from config.settings import NEWS_API_KEY
import logging

logger = logging.getLogger(__name__)

def fetch_and_produce():
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    url = f"https://newsapi.org/v2/everything?q=stock+market&apiKey={NEWS_API_KEY}"
    response = requests.get(url, timeout=10)
    articles = response.json().get("articles", [])
    
    for article in articles[:5]:
        message = {"headline": article["title"]}
        producer.send("market_feed", value=message)
        print(f"Produced: {article['title'][:60]}...")
    
    producer.flush()
    print("All messages sent to Kafka.")

if __name__ == "__main__":
    fetch_and_produce()