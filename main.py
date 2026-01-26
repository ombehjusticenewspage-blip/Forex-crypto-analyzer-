import os, time, asyncio, requests, numpy as np, pandas as pd, ta
from textblob import TextBlob
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes


TELEGRAM_TOKEN ="8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
NEWS_API_KEY = "ca1acbf0cedb4488b130c59252891c5e"
MONITOR_INTERVAL = 90  

CRYPTOS = {"BTCUSDT":"bitcoin", "ETHUSDT":"ethereum", "BNBUSDT":"binancecoin", "XRPUSDT":"ripple", "SOLUSDT":"solana"}
FOREX = {"EURUSD":"EURUSD=X", "GBPUSD":"GBPUSD=X", "USDJPY":"USDJPY=X", "USDCHF":"USDCHF=X", "XAUUSD":"XAUUSD=X"}
ALL_ASSETS = dict(**CRYPTOS, **FOREX)

class DataSource:
    @staticmethod
    def fetch_crypto(symbol):
        if symbol not in CRYPTOS: return pd.DataFrame()
        coin_id = CRYPTOS[symbol]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency":"usd","days":2,"interval":"minute"}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            prices = data.get("prices",[])
            if len(prices)<50: return pd.DataFrame()
            df = pd.DataFrame(prices, columns=["time","c"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["o"] = df["c"].shift(1)
            df["h"] = df[["o","c"]].max(axis=1)
            df["l"] = df[["o","c"]].min(axis=1)
            return df.dropna().reset_index(drop=True)
        except Exception as e:
            print("[Crypto fetch error]", e); return pd.DataFrame()
    @staticmethod
    def fetch_forex(symbol):
        if symbol not in FOREX: return pd.DataFrame()
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{FOREX[symbol]}"
        params = {"interval":"5m","range":"1d"}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            t = j['chart']['result'][0]['timestamp']
            prices = j['chart']['result'][0]['indicators']['quote'][0]['close']
            highs = j['chart']['result'][0]['indicators']['quote'][0]['high']
            lows  = j['chart']['result'][0]['indicators']['quote'][0]['low']
            df = pd.DataFrame({"time":pd.to_datetime(t, unit="s"), "c":prices, "h":highs, "l":lows})
            df["o"] = df["c"].shift(1)
            return df.dropna().reset_index(drop=True)
        except Exception as e:
            print("[Forex fetch error]", e); return pd.DataFrame()

    @staticmethod
    def fetch(symbol):
        if symbol in CRYPTOS: return DataSource.fetch_crypto(symbol)
        elif symbol in FOREX:  return DataSource.fetch_forex(symbol)
        else: return pd.DataFrame()

class NewsSentiment:
    @staticmethod
    def score(symbol):
        name = symbol.replace("USDT","").replace("=","")
        url = "https://newsapi.org/v2/everything"
        params = {"q":name,"language":"en","sortBy":"publishedAt","apiKey":NEWS_API_KEY,"pageSize":5}
        try:
            r = requests.get(url, params=params, timeout=7).json()
            articles = r.get("articles",[])
            if not articles: return 0
            return sum(TextBlob(a["title"]).sentiment.polarity for a in articles)/len(articles)
        except Exception as e:
            print("[News sentiment error]", e); return 0

class Indicators:
    @staticmethod
    def enrich(df):
        try:
            df["RSI"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
            df["MACD"] = ta.trend.MACD(df["c"]).macd_diff()
            df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range()
            df["EMA20"] = ta.trend.EMAIndicator(df["c"],20).ema_indicator()
            df["BB"] = ta.volatility.BollingerBands(df["c"],20).bollinger_hband() - ta.volatility.BollingerBands(df["c"],20).bollinger_lband()
            return df.dropna().reset_index(drop=True)
        except Exception as e:
            print("[Indicator error]", e); return pd.DataFrame()
        
class SignalEngine:
    @staticmethod
    def generate(df, sentiment):
        
        prev = df.iloc[-2]
        last = df.iloc[-1]
        signals = []
        if last["RSI"] < 35 and last["MACD"] > 0: signals.append("BUY")
        elif last["RSI"] > 70 and last["MACD"] < 0: signals.append("SELL")
        if last["c"] > last["EMA20"]: signals.append("BUY")
        elif last["c"] < last["EMA20"]: signals.append("SELL")
        
        s_bias = "BUY" if sentiment > 0.05 else "SELL" if sentiment < -0.05 else ""
        if s_bias: signals.append(s_bias)
        
        direction = "BUY" if signals.count("BUY") > signals.count("SELL") else "SELL" if signals.count("SELL") > signals.count("BUY") else "HOLD"
        confidence = min(max(abs(last["RSI"]-50)/50 + abs(sentiment),0),1)
        atr = last["ATR"]
        price = last["c"]
        sl = price - atr if direction=="BUY" else price + atr
        tp = price + atr*2 if direction=="BUY" else price - atr*2
        return dict(direction=direction,confidence=confidence,price=price,sl=sl,tp=tp,atr=atr)

class TradeMonitor:
    open_trades = {}
    @classmethod
    async def watch(cls, asset, user_id, context):
        if asset not in cls.open_trades: return
        plan = cls.open_trades[asset]
        entry, sl, tp, direction = plan["price"], plan["sl"], plan["tp"], plan["direction"]
        while True:
            await asyncio.sleep(MONITOR_INTERVAL)
            df = DataSource.fetch(asset)
            if df.empty: continue
            price = df.iloc[-1]["c"]
            if direction == "BUY" and price < sl:
                await context.bot.send_message(user_id, f"❗ {asset}: Price fell below stop-loss. Consider closing your BUY trade. Last Price: {round(price,5)}")
                break
            if direction == "SELL" and price > sl:
                await context.bot.send_message(user_id, f"❗ {asset}: Price rose above stop-loss. Consider closing your SELL trade. Last Price: {round(price,5)}")
                break
            if direction == "BUY" and price > tp:
                await context.bot.send_message(user_id, f"✅ {asset}: Take-profit hit! Close your trade. Last Price: {round(price,5)}")
                break
            if direction == "SELL" and price < tp:
                await context.bot.send_message(user_id, f"✅ {asset}: Take-profit hit! Close your trade. Last Price: {round(price,5)}")
                break

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(k, callback_data=k)] for k in ALL_ASSETS]
    await update.message.reply_text("Select Asset (Crypto/Forex):", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    await q.edit_message_text(f"Analyzing {asset} ...")
    df = DataSource.fetch(asset)
    if df.empty: await q.edit_message_text("❌ Market data unavailable"); return
    df = Indicators.enrich(df)
    if df.empty: await q.edit_message_text("❌ Market data enrichment failed."); return
    sentiment = NewsSentiment.score(asset)
    plan = SignalEngine.generate(df, sentiment)
    msg = (f"🧠 Trade Plan\n\nAsset: {asset}\nDirection: {plan['direction']}\nEntry: {round(plan['price'],5)}\n"
           f"SL: {round(plan['sl'],5)}\nTP: {round(plan['tp'],5)}"
           f"\nForecast Confidence: {round(plan['confidence']*100,2)}%\n(sentiment={round(sentiment,3)})")
    await q.edit_message_text(msg)
    
    user_id = q.from_user.id
    if plan["direction"] in ("BUY","SELL"):
        TradeMonitor.open_trades[asset] = {**plan, "user_id": user_id}
        asyncio.create_task(TradeMonitor.watch(asset, user_id, context))

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()