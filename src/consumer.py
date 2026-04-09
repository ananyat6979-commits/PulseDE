from kafka import KafkaConsumer
from transformers import pipeline
import json
import logging

logger = logging.getLogger(__name__)

def consume_and_analyze():
    consumer = KafkaConsumer(
        "market_feed",
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    
    print("Consumer waiting for messages...\n")
    
    for message in consumer:
        headline = message.value["headline"]
        result = sentiment_pipeline([headline])[0]
        print(f"Headline: {headline[:60]}...")
        print(f"Sentiment: {result['label']} ({round(result['score'] * 100, 2)}%)\n")

if __name__ == "__main__":
    consume_and_analyze()