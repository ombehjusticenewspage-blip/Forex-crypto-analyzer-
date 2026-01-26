import os, asyncio, requests, numpy as np, pandas as pd, ta
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = "8516981161:AAFbLbt8YDXk3qAXsd1t66ZL4IGP8Zxxmkc"
NEWS_API = "332bf45035354091b59f1f64601e2e11"
MODEL_PATH = "ai_model_portfolio.h5"

COINS = {
    "BTCUSDT":"bitcoin","ETHUSDT":"ethereum","BNBUSDT":"binancecoin","SOLUSDT":"solana",
    "XRPUSDT":"ripple","ADAUSDT":"cardano","DOGEUSDT":"dogecoin","AVAXUSDT":"avalanche-2",
    "DOTUSDT":"polkadot","TRXUSDT":"tron","MATICUSDT":"matic-network","LINKUSDT":"chainlink",
    "LTCUSDT":"litecoin","ATOMUSDT":"cosmos","ETCUSDT":"ethereum-classic",
    "FILUSDT":"filecoin","NEARUSDT":"near","APTUSDT":"aptos","ARBUSDT":"arbitrum",
    "OPUSDT":"optimism","SUIUSDT":"sui","AAVEUSDT":"aave","UNIUSDT":"uniswap",
    "INJUSDT":"injective-protocol","RNDRUSDT":"render-token","IMXUSDT":"immutable-x",
    "STXUSDT":"blockstack","HBARUSDT":"hedera-hashgraph","ICPUSDT":"internet-computer",
    "VETUSDT":"vechain","THETAUSDT":"theta-token","EGLDUSDT":"elrond-erd-2",
    "ALGOUSDT":"algorand","FLOWUSDT":"flow","GRTUSDT":"the-graph",
    "KAVAUSDT":"kava","FTMUSDT":"fantom","RUNEUSDT":"thorchain",
    "MINAUSDT":"mina-protocol","DYDXUSDT":"dydx",
    "WAVESUSDT":"waves","ZILUSDT":"zilliqa","ENJUSDT":"enjincoin",
    "SANDUSDT":"the-sandbox","MANAUSDT":"decentraland","AXSUSDT":"axie-infinity",
    "CHZUSDT":"chiliz","CAKEUSDT":"pancakeswap-token"
}

def fetch_coingecko(symbol):
    coin_id = COINS[symbol]
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":7,"interval":"minute"}
    r = requests.get(url, params=params, timeout=15).json()
    prices = r.get("prices",[])
    volumes = r.get("total_volumes",[])
    if len(prices) < 100:
        return pd.DataFrame()
    df = pd.DataFrame(prices, columns=["time","c"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["v"] = [v[1] for v in volumes[:len(df)]]
    df["o"] = df["c"].shift(1)
    df["h"] = df[["o","c"]].max(axis=1)
    df["l"] = df[["o","c"]].min(axis=1)
    return df.dropna().reset_index(drop=True)

def news_sentiment(symbol):
    try:
        q = symbol.replace("USDT","")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q":q,"language":"en","apiKey":NEWS_API,"pageSize":5},
            timeout=10
        ).json()
        articles = r.get("articles",[])
        if not articles:
            return 0
        return sum(TextBlob(a["title"]).sentiment.polarity for a in articles) / len(articles)
    except:
        return 0

def enrich(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
    df["EMA"] = ta.trend.EMAIndicator(df["c"],20).ema_indicator()
    df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range()
    return df.dropna()

class MarketAI:
    def __init__(self, window=30):
        self.window = window
        self.scaler = MinMaxScaler()
        self.model = self.load()

    def load(self):
        if os.path.exists(MODEL_PATH):
            return load_model(MODEL_PATH)
        m = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.window,5)),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation="sigmoid")
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy")
        return m

    def features(self, df):
        df["r"] = df["c"].pct_change()
        df["v"] = df["r"].rolling(10).std()
        df["ed"] = df["c"] - df["EMA"]
        return df.dropna()[["r","v","RSI","ed","c"]]

    def train(self, df):
        f = self.features(df)
        if len(f) <= self.window:
            return
        s = self.scaler.fit_transform(f)
        X, y = [], []
        for i in range(self.window, len(s)-1):
            X.append(s[i-self.window:i])
            y.append(1 if f["c"].iloc[i+1] > f["c"].iloc[i] else 0)
        self.model.fit(np.array(X), np.array(y), epochs=3, batch_size=8, verbose=0)
        self.model.save(MODEL_PATH)

    def predict(self, df):
        f = self.features(df)
        if len(f) < self.window:
            return 0.5
        s = self.scaler.fit_transform(f)
        return float(self.model.predict(np.array([s[-self.window:]]), verbose=0)[0][0])

class Trader:
    def decide(self, prob, news):
        score = abs(prob-0.5)*2 + abs(news)
        if score < 0.6:
            return "HOLD", score
        return ("BUY" if prob > 0.5 else "SELL"), score

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(k, callback_data=k)] for k in COINS]
    await update.message.reply_text("Select Crypto Asset", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    await q.edit_message_text(f"🔍 Analyzing {asset}…")

    df = await asyncio.to_thread(fetch_coingecko, asset)
    if df.empty:
        await q.edit_message_text("❌ Market data unavailable")
        return

    df = await asyncio.to_thread(enrich, df)

    ai = MarketAI()
    await asyncio.to_thread(ai.train, df)
    prob = await asyncio.to_thread(ai.predict, df)
    news = await asyncio.to_thread(news_sentiment, asset)

    decision, confidence = Trader().decide(prob, news)

    price = df["c"].iloc[-1]
    atr = df["ATR"].iloc[-1]
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

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()