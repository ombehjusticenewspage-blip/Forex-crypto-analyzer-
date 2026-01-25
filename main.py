import os, requests, numpy as np, pandas as pd, ta
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = "8516981161:AAFbLbt8YDXk3qAXsd1t66ZL4IGP8Zxxmkc"
NEWS_API = "332bf45035354091b59f1f64601e2e11"
BINANCE_API_KEY = "gD3Prl4zcqEsx8sXvC09XAxlJXGDqMNZ28j6ol43x0mTbtO88XzuHWUHACtMoUto"

MODEL_PATH = "ai_model_portfolio.h5"

CRYPTO = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
    "AVAXUSDT","DOTUSDT","TRXUSDT","MATICUSDT","LINKUSDT","LTCUSDT","ATOMUSDT",
    "ETCUSDT","FILUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","SUIUSDT",
    "AAVEUSDT","UNIUSDT","INJUSDT","RNDRUSDT","IMXUSDT","STXUSDT","HBARUSDT",
    "ICPUSDT","VETUSDT","THETAUSDT","EGLDUSDT","ALGOUSDT","FLOWUSDT",
    "GRTUSDT","KAVAUSDT","FTMUSDT","RUNEUSDT","MINAUSDT","DYDXUSDT",
    "WAVESUSDT","ZILUSDT","ENJUSDT","SANDUSDT","MANAUSDT","AXSUSDT",
    "CHZUSDT","CAKEUSDT"
]

def fetch_binance(symbol, interval="15m", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 100:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "time","o","h","l","c","v",
            "ct","qav","trades","tb","tq","ignore"
        ])
        df = df[["time","o","h","l","c","v"]]
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        return df.reset_index(drop=True)
    except:
        return pd.DataFrame()

def news_sentiment(symbol):
    try:
        q = symbol.replace("USDT","")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": q, "language": "en", "apiKey": NEWS_API, "pageSize": 5},
            timeout=10
        ).json()
        articles = r.get("articles", [])
        if not articles:
            return 0
        return sum(TextBlob(a["title"]).sentiment.polarity for a in articles) / len(articles)
    except:
        return 0

def enrich(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["c"], 14).rsi()
    df["EMA"] = ta.trend.EMAIndicator(df["c"], 20).ema_indicator()
    df["ATR"] = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"], 14).average_true_range()
    return df.dropna()

class MarketAI:
    def __init__(self, window=30):
        self.window = window
        self.scaler = MinMaxScaler()
        self.model = self.load_or_create()

    def load_or_create(self):
        if os.path.exists(MODEL_PATH):
            return load_model(MODEL_PATH)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.window,5)),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def features(self, df):
        df["r"] = df["c"].pct_change()
        df["v"] = df["r"].rolling(10).std()
        df["ed"] = df["c"] - df["EMA"]
        df = df.dropna()
        return df[["r","v","RSI","ed","c"]]

    def train(self, df):
        f = self.features(df)
        if len(f) <= self.window:
            return
        s = self.scaler.fit_transform(f)
        X, y = [], []
        for i in range(self.window, len(s)-1):
            X.append(s[i-self.window:i])
            y.append(1 if f["c"].iloc[i+1] > f["c"].iloc[i] else 0)
        X, y = np.array(X), np.array(y)
        self.model.fit(X, y, epochs=3, batch_size=8, verbose=0)
        self.model.save(MODEL_PATH)

    def predict(self, df):
        f = self.features(df)
        if len(f) < self.window:
            return 0.5
        s = self.scaler.fit_transform(f)
        X = np.array([s[-self.window:]])
        return float(self.model.predict(X, verbose=0)[0][0])

class Trader:
    def decide(self, prob, news):
        score = abs(prob - 0.5) * 2 + abs(news)
        if score < 0.35:
            return "HOLD", score
        return ("BUY" if prob > 0.5 else "SELL"), score

AI_ENGINE = MarketAI()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(a, callback_data=a)] for a in CRYPTO]
    await update.message.reply_text("Select Crypto Asset", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    await q.edit_message_text(f"🔍 Analyzing {asset}…")

    df = fetch_binance(asset)
    if df.empty:
        await q.edit_message_text("❌ Market data unavailable")
        return

    df = enrich(df)

    AI_ENGINE.train(df)
    prob = AI_ENGINE.predict(df)
    news = news_sentiment(asset)

    decision, confidence = Trader().decide(prob, news)

    price = df["c"].iloc[-1]
    atr = df["ATR"].iloc[-1]

    sl = price - atr if decision == "BUY" else price + atr
    tp = price + atr*2 if decision == "BUY" else price - atr*2

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

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()