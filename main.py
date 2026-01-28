import asyncio
import requests
import pandas as pd
import ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
MONITOR_INTERVAL = 90

CRYPTOS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"
]

FOREX = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","XAUUSD"
]

ALL_ASSETS = CRYPTOS + FOREX

class DataSource:

    def crypto(symbol):
        url = "https://data.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "5m", "limit": 200}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data, columns=[
            "t","o","h","l","c","v",
            "_","_","_","_","_","_"
        ])
        df["c"] = df["c"].astype(float)
        df["h"] = df["h"].astype(float)
        df["l"] = df["l"].astype(float)
        return df[["c","h","l"]]

    def forex(symbol):
        base = symbol[:3]
        quote = symbol[3:]
        url = f"https://api.frankfurter.app/{base}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        rate = r.json()["rates"].get(quote)
        if not rate:
            return pd.DataFrame()
        df = pd.DataFrame({"c":[rate]*50,"h":[rate]*50,"l":[rate]*50})
        return df

    def fetch(symbol):
        try:
            if symbol in CRYPTOS:
                return DataSource.crypto(symbol)
            else:
                return DataSource.forex(symbol)
        except Exception as e:
            print("[DATA ERROR]", symbol, e)
            return pd.DataFrame()

class Indicators:
    def enrich(df):
        df["RSI"] = ta.momentum.RSIIndicator(df["c"]).rsi()
        df["MACD"] = ta.trend.MACD(df["c"]).macd_diff()
        df["EMA"] = ta.trend.EMAIndicator(df["c"]).ema_indicator()
        df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"]).average_true_range()
        return df.dropna()

class SignalEngine:
    def generate(df):
        last = df.iloc[-1]
        if last.RSI < 35 and last.MACD > 0:
            direction = "BUY"
        elif last.RSI > 70 and last.MACD < 0:
            direction = "SELL"
        else:
            direction = "HOLD"

        price = last.c
        atr = last.ATR
        return {
            "direction": direction,
            "price": price,
            "sl": price - atr if direction=="BUY" else price + atr,
            "tp": price + atr*2 if direction=="BUY" else price - atr*2
        }

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(a, callback_data=a)] for a in ALL_ASSETS]
    await update.message.reply_text("Select asset:", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    await q.edit_message_text(f"Analyzing {asset}…")

    df = DataSource.fetch(asset)
    if df.empty:
        await q.edit_message_text("❌ No market data available")
        return

    df = Indicators.enrich(df)
    plan = SignalEngine.generate(df)

    msg = (
        f"📊 {asset}\n"
        f"Direction: {plan['direction']}\n"
        f"Entry: {round(plan['price'],5)}\n"
        f"SL: {round(plan['sl'],5)}\n"
        f"TP: {round(plan['tp'],5)}"
    )
    await q.edit_message_text(msg)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    print("Bot running")
    app.run_polling()