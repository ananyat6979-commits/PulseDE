import json
import http.server
import socketserver
import os

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PulseDE: Market Intelligence</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #0f0f0f; color: #fff; padding: 40px; }}
        h1 {{ color: #00ff88; }}
        .card {{ background: #1a1a1a; border-radius: 8px; padding: 20px; margin: 10px 0; border-left: 4px solid #333; }}
        .positive {{ border-left-color: #00ff88; }}
        .negative {{ border-left-color: #ff4444; }}
        .neutral {{ border-left-color: #ffaa00; }}
        .sentiment {{ font-size: 12px; text-transform: uppercase; letter-spacing: 2px; }}
        .confidence {{ color: #888; font-size: 12px; }}
        .timestamp {{ color: #555; font-size: 11px; }}
    </style>
</head>
<body>
    <h1> PulseDE Market Intelligence</h1>
    <p style="color:#888">Live sentiment analysis on market headlines</p>
    {cards}
</body>
</html>
"""

CARD = """
<div class="card {sentiment}">
    <div class="sentiment">{sentiment} {emoji}</div>
    <p>{headline}</p>
    <div class="confidence">Confidence: {confidence}%</div>
    <div class="timestamp">{timestamp}</div>
</div>
"""

def build_dashboard():
    with open("data/results.json") as f:
        results = json.load(f)
    
    emojis = {"positive": "[+]", "negative": "[-]", "neutral": "[~]"}
    cards = ""
    for r in results:
        cards += CARD.format(
            sentiment=r["sentiment"],
            emoji=emojis.get(r["sentiment"], ""),
            headline=r["headline"],
            confidence=r["confidence"],
            timestamp=r["timestamp"]
        )
    
    with open("data/dashboard.html", "w", encoding="utf-8") as f:
        f.write(HTML.format(cards=cards))
    
    print("Dashboard built. Opening on http://localhost:8080")
    os.chdir("data")
    with socketserver.TCPServer(("", 8080), http.server.SimpleHTTPRequestHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    build_dashboard()