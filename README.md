# PulseDE: Real-Time Market Intelligence Pipeline

A real-time data engineering pipeline that fetches live financial headlines, runs them through a FinBERT NLP model for sentiment analysis, persists structured results, and serves a live dashboard.

## Architecture

NewsAPI → fetch_headlines() → FinBERT Sentiment Model → JSON Storage → HTML Dashboard
→ Log File

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Data Ingestion | NewsAPI + requests | Real-time financial headline feed |
| Sentiment Analysis | FinBERT (ProsusAI) | BERT model fine-tuned on financial text |
| Orchestration | schedule | Lightweight hourly pipeline automation |
| Storage | JSON with timestamps | Structured, queryable output artifacts |
| Observability | Python logging | Persistent logs for monitoring and debugging |
| Dashboard | Python HTTP server + HTML | Live visualization of sentiment results |

## Why FinBERT over general sentiment models?

General sentiment models (VADER, TextBlob) are trained on social media and reviews. Financial text has domain-specific language where words like "volatile", "correction", and "bearish" carry precise meanings. FinBERT is fine-tuned on financial news and SEC filings, making it significantly more accurate for market sentiment classification.

## Project Structure
PulseDE/
├── src/
│   ├── main.py          # Core pipeline: ingest → analyze → persist
│   ├── scheduler.py     # Hourly automation
│   └── dashboard.py     # HTML dashboard server
├── config/
│   └── settings.py      # Centralized configuration
├── data/
│   └── results.json     # Sentiment output artifacts
├── logs/
│   └── pulsede.log      # Pipeline execution logs
├── .env                 # API keys (never committed)
└── requirements.txt

## How to Run
```bash
# Clone and setup
git clone https://github.com/ananyat6979-commits/PulseDE.git
cd PulseDE
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Add your NewsAPI key to .env
echo "NEWS_API_KEY=your_key_here" > .env

# Run once
python -m src.main

# Run on schedule (every hour)
python -m src.scheduler

# Launch dashboard
python -m src.dashboard
# Open http://localhost:8080/dashboard.html
```

## Output Sample
```json
{
    "timestamp": "2026-04-05T18:00:26",
    "headline": "'Big Short' investor Michael Burry warns stocks are long overdue for a crash",
    "sentiment": "negative",
    "confidence": 95.9
}
```

## Key Engineering Decisions

- **Error handling at every external call** — network failures return empty lists, pipeline continues
- **Secrets never committed** — .env excluded via .gitignore
- **Dual logging** — stdout for development, file for production monitoring
- **Modular design** — ingest, analyze, store, visualize are fully separated functions