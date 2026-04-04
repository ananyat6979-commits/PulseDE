from dotenv import load_dotenv
import os

load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "market_feed")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")