import asyncio
import aiohttp
import pandas as pd
import ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
TWELVEDATA_KEY = "ca1acbf0cedb4488b130c59252891c5e"
ALPHAVANTAGE_KEY = "EOGVA134GOOP2UMU"

MIN_CANDLES = 120
USER_AGENT = "crypto-analyzer/5.0"

CRYPTOS = {
    "BTCUSDT":"BTC/USD","ETHUSDT":"ETH/USD","BNBUSDT":"BNB/USD",
    "XRPUSDT":"XRP/USD","SOLUSDT":"SOL/USD","ADAUSDT":"ADA/USD",
    "DOGEUSDT":"DOGE/USD","AVAXUSDT":"AVAX/USD","DOTUSDT":"DOT/USD",
    "MATICUSDT":"MATIC/USD","LTCUSDT":"LTC/USD","LINKUSDT":"LINK/USD",
    "TRXUSDT":"TRX/USD","ATOMUSDT":"ATOM/USD","UNIUSDT":"UNI/USD"
}

async def fetch_data(session, symbol):
    try:
        async with session.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": CRYPTOS[symbol],
                "interval": "15min",
                "outputsize": 500,
                "apikey": TWELVEDATA_KEY
            },
            headers={"User-Agent": USER_AGENT},
            timeout=12
        ) as r:
            j = await r.json()
            if "values" not in j:
                return pd.DataFrame()
            rows = [{
                "c": float(v["close"]),
                "h": float(v["high"]),
                "l": float(v["low"]),
                "v": float(v.get("volume",0))
            } for v in reversed(j["values"])]
            return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def enrich(df):
    if df.empty or len(df) < MIN_CANDLES:
        return pd.DataFrame()
    df["RSI"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
    df["MACD"] = ta.trend.MACD(df["c"]).macd_diff()
    df["EMA50"] = ta.trend.EMAIndicator(df["c"],50).ema_indicator()
    df["EMA200"] = ta.trend.EMAIndicator(df["c"],200).ema_indicator()
    df["ADX"] = ta.trend.ADXIndicator(df["h"],df["l"],df["c"]).adx()
    df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"]).average_true_range()
    df["VWAP"] = (df["v"]*(df["h"]+df["l"]+df["c"])/3).cumsum() / df["v"].cumsum()
    return df.dropna()

def score_signal(df):
    last = df.iloc[-1]
    score = 0
    score += 2 if last["EMA50"] > last["EMA200"] else -2
    score += 1 if last["RSI"] > 55 else -1 if last["RSI"] < 45 else 0
    score += 1 if last["MACD"] > 0 else -1
    score += 1 if last["ADX"] > 20 else 0
    score += 1 if last["c"] > last["VWAP"] else -1
    direction = "BUY" if score >= 3 else "SELL" if score <= -3 else None
    if not direction:
        return None
    price = float(last["c"])
    atr = float(last["ATR"])
    sl = price - atr*1.5 if direction=="BUY" else price + atr*1.5
    tp = price + atr*3 if direction=="BUY" else price - atr*3
    confidence = min(96, 60 + abs(score)*6)
    return {
        "dir": direction,
        "price": price,
        "sl": sl,
        "tp": tp,
        "score": score,
        "conf": confidence
    }

async def scan_market():
    results = []
    async with aiohttp.ClientSession() as session:
        for symbol in CRYPTOS:
            df = await fetch_data(session, symbol)
            df = enrich(df)
            if df.empty:
                continue
            signal = score_signal(df)
            if signal:
                results.append((symbol, signal))
    results.sort(key=lambda x: abs(x[1]["score"]), reverse=True)
    return results[:3]

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📡 Strong crypto signals activated, top 3 loading...")
    signals = await scan_market()
    if not signals:
        await update.message.reply_text("⚠️ No high-probability crypto setups right now")
        return
    msg = "🚀 TOP CRYPTO SIGNALS\n\n"
    for i,(sym,s) in enumerate(signals,1):
        msg += (
            f"{i}. {sym}\n"
            f"Direction: {s['dir']}\n"
            f"Entry: {round(s['price'],5)}\n"
            f"SL: {round(s['sl'],5)}\n"
            f"TP: {round(s['tp'],5)}\n"
            f"Confidence: {s['conf']}%\n\n"
        )
    await update.message.reply_text(msg)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.run_polling()