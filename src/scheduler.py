import schedule
import time
from src.main import fetch_headlines, analyze_sentiment, save_results

def run_pipeline():
    print("\n--- PulseDE Pipeline Running ---")
    headlines = fetch_headlines()
    results = analyze_sentiment(headlines)
    save_results(headlines, results)
    print("--- Pipeline Complete ---\n")

if __name__ == "__main__":
    print("PulseDE Scheduler Started. Running every hour.")
    run_pipeline()  # run once immediately on start
    schedule.every(1).hours.do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(60)