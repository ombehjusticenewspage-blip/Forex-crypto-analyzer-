import os, requests, numpy as np, pandas as pd, ta, pytz
from datetime import datetime
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
NEWS_API = "332bf45035354091b59f1f64601e2e11"
TWELVE_API = "ca1acbf0cedb4488b130c59252891c5e"

MODEL_PATH = "ai_model_portfolio.h5"

CRYPTO = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"]
FOREX = ["EURUSD","GBPUSD","USDJPY","XAUUSD"]
UTC = pytz.UTC

portfolio = {}

def fetch_twelvedata(symbol, interval="15min", outputsize=2000, retries=3):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_API}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10).json()
            if "values" not in r or not r["values"]: 
                continue
            df = pd.DataFrame(r["values"])
            df = df.rename(columns={"datetime":"time","open":"o","high":"h","low":"l","close":"c","volume":"v"})
            df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].apply(pd.to_numeric, errors="coerce")
            df = df.dropna().sort_values("time").reset_index(drop=True)
            if len(df) >= 30:
                return df
        except:
            continue
    # fallback for forex
    if "/" in symbol and symbol.split("/")[0] in ["EUR","GBP","USD","XAU"]:
        alt_intervals = ["5min","1min"]
        for alt in alt_intervals:
            try:
                params["interval"] = alt
                r = requests.get(url, params=params, timeout=10).json()
                if "values" not in r or not r["values"]:
                    continue
                df = pd.DataFrame(r["values"])
                df = df.rename(columns={"datetime":"time","open":"o","high":"h","low":"l","close":"c","volume":"v"})
                df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].apply(pd.to_numeric, errors="coerce")
                df = df.dropna().sort_values("time").reset_index(drop=True)
                if len(df) >= 30:
                    return df
            except:
                continue
    return pd.DataFrame(columns=["o","h","l","c","v"])

def crypto_data(sym, tf="15min"):
    symbol_td = sym[:3] + "/" + sym[3:]
    return fetch_twelvedata(symbol_td, tf)

def forex_data(pair, tf="15min"):
    symbol_td = pair[:3] + "/" + pair[3:]
    return fetch_twelvedata(symbol_td, tf)

def news_sentiment(symbol):
    try:
        q = symbol.replace("USDT","")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": q, "language": "en", "apiKey": NEWS_API,"pageSize":5},
            timeout=10
        ).json()
        articles = r.get("articles",[])
        if not articles: return 0
        score = sum(TextBlob(a.get("title","")).sentiment.polarity for a in articles)
        return score / len(articles)
    except:
        return 0

def enrich(df):
    if len(df) < 20: return df
    df["EMA20"] = ta.trend.EMAIndicator(df["c"],20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["c"],50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
    macd = ta.trend.MACD(df["c"])
    df["MACD"] = macd.macd()
    df["MS"] = macd.macd_signal()
    df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range()
    return df.dropna()

class MarketAI:
    def __init__(self, window=30):
        self.window = window
        self.scaler = MinMaxScaler()
        self.model = self.load_or_create()
    def load_or_create(self):
        if os.path.exists(MODEL_PATH):
            try: return load_model(MODEL_PATH)
            except: os.remove(MODEL_PATH)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.window,5)),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model
    def features(self, df):
        df = df.copy()
        df["r"] = df["c"].pct_change()
        df["v"] = df["r"].rolling(10).std()
        df["rsi"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
        df["ema"] = ta.trend.EMAIndicator(df["c"],20).ema_indicator()
        df["ed"] = df["c"] - df["ema"]
        df = df.dropna()
        return df[["r","v","rsi","ed","c"]]
    def prepare(self, df):
        f = self.features(df)
        if len(f) <= self.window: return None, None
        s = self.scaler.fit_transform(f)
        X, y = [], []
        for i in range(self.window, len(s)-1):
            X.append(s[i-self.window:i])
            y.append(1 if f["c"].iloc[i+1] > f["c"].iloc[i] else 0)
        return np.array(X), np.array(y)
    def train_daily(self, df):
        X, y = self.prepare(df)
        if X is None or len(X)==0: return
        self.model.fit(X, y, epochs=3, batch_size=8, verbose=0)
        self.model.save(MODEL_PATH)
    def predict(self, df):
        f = self.features(df)
        if len(f) < self.window: return None
        s = self.scaler.fit_transform(f)
        X = np.array([s[-self.window:]])
        return float(self.model.predict(X, verbose=0)[0][0])

class RLTrader:
    def decide(self, prob, news):
        score = abs(prob-0.5)*2 + abs(news)
        if score < 0.6: return "NO TRADE", score
        return ("BUY" if prob>0.5 else "SELL"), score

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(a, callback_data=a)] for a in sorted(set(CRYPTO+FOREX))]
    await update.message.reply_text("Select Asset for AI Trading", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    is_crypto = asset in CRYPTO
    await q.edit_message_text(f"🔍 Analyzing {asset}...\nPlease wait ⏳")
    df = crypto_data(asset) if is_crypto else forex_data(asset)
    df = enrich(df)
    # Removed the "not enough data" check completely
    ai = MarketAI()
    ai.train_daily(df)
    prob = ai.predict(df)
    if prob is None: prob = 0.5
    news = news_sentiment(asset)
    decision, confidence = RLTrader().decide(prob, news)
    if decision == "NO TRADE": decision = "HOLD"
    price = df["c"].iloc[-1] if len(df) else 0
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 0
    sl = price - atr if decision=="BUY" else price + atr
    tp = price + atr*2 if decision=="BUY" else price - atr*2
    await q.edit_message_text(
        f"🧠 AI Hedge Fund Trade Plan\n\n"
        f"Asset: {asset}\n"
        f"Direction: {decision}\n"
        f"Entry: {round(price,5)}\n"
        f"SL: {round(sl,5)}\n"
        f"TP: {round(tp,5)}\n\n"
        f"Probability: {round(prob,3)}\n"
        f"Confidence: {round(confidence*100,1)}%"
    )

if __name__=="__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    print("Bot running")
    app.run_polling()