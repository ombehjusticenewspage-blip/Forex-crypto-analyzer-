
import os
import requests
import sys
from binance.client import Client

def test_binance():
BINANCE_API_KEY="DmIS0QxaE6XsUnVCgpewCV4KdFuxFjmQhfhm2YRIiyPQsRQ6vcl1QTp1tWpxLN3Z"
BINANCE_API_SECRET="KRDBzIVMIVa0kG0vdfu4q8d7CBB6bOYi9e7tWBKB0IAbxPwbkCavlmpNBGaxmt8J"

    if not key or not secret:
        print("[Binance] SKIPPED: BINANCE_API_KEY/SECRET not set")
        return
    try:
        c = Client(key, secret)
        kl = c.get_klines(symbol="BTCUSDT", interval="5m", limit=5)
        print(f"[Binance] OK: returned {len(kl)} kline(s)")
    except Exception as e:
        print(f"[Binance] ERROR: {e}")

def test_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency":"usd","days":1,"interval":"hourly"}
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"[CoinGecko] HTTP {r.status_code}")
        r.raise_for_status()
        j = r.json()
        prices = j.get("prices", [])
        print(f"[CoinGecko] OK: prices length = {len(prices)}")
    except Exception as e:
        print(f"[CoinGecko] ERROR: {e}")

def test_yahoo():
    
    url = "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X"
    params = {"interval":"5m","range":"1d"}
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"[Yahoo] HTTP {r.status_code}")
        r.raise_for_status()
        j = r.json()
        ok = 'chart' in j and 'result' in j['chart'] and bool(j['chart']['result'])
        if ok:
            res = j['chart']['result'][0]
            ts = res.get("timestamp", [])
            quotes = res['indicators']['quote'][0].get("close", []) if res.get('indicators') else []
            print(f"[Yahoo] OK: timestamps={len(ts)}, closes={len(quotes)}")
        else:
            print("[Yahoo] ERROR: unexpected JSON structure")
    except Exception as e:
        print(f"[Yahoo] ERROR: {e}")

def test_newsapi():
    key = os.getenv("NEWS_API_KEY", "")
    if not key:
        print("[NewsAPI] SKIPPED: NEWS_API_KEY not set")
        return
    url = "https://newsapi.org/v2/everything"
    params = {"q":"bitcoin","language":"en","pageSize":1,"apiKey":key}
    try:
        r = requests.get(url, params=params, timeout=7)
        print(f"[NewsAPI] HTTP {r.status_code}")
        r.raise_for_status()
        j = r.json()
        total = j.get("totalResults", None)
        articles = len(j.get("articles", []))
        print(f"[NewsAPI] OK: totalResults={total}, articles returned={articles}")
    except Exception as e:
        print(f"[NewsAPI] ERROR: {e}")

def test_telegram():
TELEGRAM_TOKEN="8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
    if not token:
        print("[Telegram] SKIPPED: TELEGRAM_TOKEN not set")
        return
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        r = requests.get(url, timeout=7)
        print(f"[Telegram] HTTP {r.status_code}")
        r.raise_for_status()
        j = r.json()
        if j.get("ok"):
            u = j.get("result", {}).get("username", "<unknown>")
            print(f"[Telegram] OK: bot username = @{u}")
        else:
            print(f"[Telegram] ERROR: API returned ok=false: {j}")
    except Exception as e:
        print(f"[Telegram] ERROR: {e}")

if __name__ == "__main__":
    print("Starting connectivity tests...\n")
    test_binance()
    test_coingecko()
    test_yahoo()
    test_newsapi()
    test_telegram()
    print("\nDone.")