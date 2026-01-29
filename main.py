import asyncio
import requests
import numpy as np
import pandas as pd
import ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
ALPHAVANTAGE_KEY = "EOGVA134GOOP2UMU"
TWELVEDATA_KEY = "ca1acbf0cedb4488b130c59252891c5e"

MONITOR_INTERVAL = 90
MIN_CANDLES = 60
USER_AGENT = "forex-crypto-analyzer/1.0"

CRYPTOS = {
    "BTCUSDT": "BTC/USD",
    "ETHUSDT": "ETH/USD",
    "BNBUSDT": "BNB/USD",
    "XRPUSDT": "XRP/USD",
    "SOLUSDT": "SOL/USD"
}

FOREX = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "XAUUSD": "XAU/USD"
}

ALL_ASSETS = dict(**CRYPTOS, **FOREX)

def fetch_twelvedata(symbol):
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": "5min",
                "outputsize": 500,
                "apikey": TWELVEDATA_KEY
            },
            timeout=12,
            headers={"User-Agent": USER_AGENT}
        )
        j = r.json()
        if "values" not in j:
            return pd.DataFrame()

        rows = []
        for v in reversed(j["values"]):
            rows.append({
                "time": pd.to_datetime(v["datetime"]),
                "o": float(v["open"]),
                "h": float(v["high"]),
                "l": float(v["low"]),
                "c": float(v["close"]),
                "v": float(v.get("volume", 0))
            })

        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def fetch_alpha_forex(pair):
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "FX_INTRADAY",
                "from_symbol": pair[:3],
                "to_symbol": pair[3:],
                "interval": "5min",
                "apikey": ALPHAVANTAGE_KEY
            },
            timeout=12,
            headers={"User-Agent": USER_AGENT}
        )
        j = r.json()
        key = next((k for k in j if "Time Series" in k), None)
        if not key:
            return pd.DataFrame()

        rows = []
        for t, v in j[key].items():
            rows.append({
                "time": pd.to_datetime(t),
                "o": float(v["1. open"]),
                "h": float(v["2. high"]),
                "l": float(v["3. low"]),
                "c": float(v["4. close"]),
                "v": 0.0
            })

        return pd.DataFrame(sorted(rows, key=lambda x: x["time"]))
    except:
        return pd.DataFrame()

def validate_df(df):
    if df.empty or len(df) < MIN_CANDLES:
        return False
    if df["c"].var() == 0:
        return False
    return True

class Indicators:
    @staticmethod
    def enrich(df):
        df = df.copy()
        df["RSI"] = ta.momentum.RSIIndicator(df["c"], 14).rsi()
        df["MACD"] = ta.trend.MACD(df["c"]).macd_diff()
        df["EMA20"] = ta.trend.EMAIndicator(df["c"], 20).ema_indicator()
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["h"], df["l"], df["c"], 14
        ).average_true_range()
        return df.dropna()

class SignalEngine:
    @staticmethod
    def generate(df):
        last = df.iloc[-1]

        buy_score = (
            last["RSI"] < 45 and
            last["MACD"] > 0 and
            last["c"] > last["EMA20"]
        )

        sell_score = (
            last["RSI"] > 55 and
            last["MACD"] < 0 and
            last["c"] < last["EMA20"]
        )

        if not buy_score and not sell_score:
            return None

        direction = "BUY" if buy_score else "SELL"
        price = float(last["c"])
        atr = float(last["ATR"])

        if atr <= 0:
            return None

        sl = price - atr if direction == "BUY" else price + atr
        tp = price + atr * 2 if direction == "BUY" else price - atr * 2

        confidence = min(
            0.8 + abs(last["RSI"] - 50) / 100,
            0.95
        )

        return {
            "direction": direction,
            "price": price,
            "sl": sl,
            "tp": tp,
            "confidence": confidence
        }

class TradeMonitor:
    open_trades = {}

    @staticmethod
    async def watch(asset, user_id, context):
        trade = TradeMonitor.open_trades.get(asset)
        if not trade:
            return

        while True:
            await asyncio.sleep(MONITOR_INTERVAL)

            df = fetch_twelvedata(ALL_ASSETS[asset]) if asset in CRYPTOS else fetch_alpha_forex(asset)
            if df.empty:
                continue

            price = float(df.iloc[-1]["c"])

            if trade["direction"] == "BUY":
                if price <= trade["sl"]:
                    await context.bot.send_message(user_id, f"❗ {asset} SL hit")
                    break
                if price >= trade["tp"]:
                    await context.bot.send_message(user_id, f"✅ {asset} TP hit")
                    break
            else:
                if price >= trade["sl"]:
                    await context.bot.send_message(user_id, f"❗ {asset} SL hit")
                    break
                if price <= trade["tp"]:
                    await context.bot.send_message(user_id, f"✅ {asset} TP hit")
                    break

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(k, callback_data=k)] for k in ALL_ASSETS]
    await update.message.reply_text(
        "Forex & Crypto Analyzer\nSelect Asset:",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    asset = q.data
    await q.edit_message_text(f"🔍 Analyzing {asset}...")

    df = fetch_twelvedata(ALL_ASSETS[asset]) if asset in CRYPTOS else fetch_alpha_forex(asset)

    if not validate_df(df):
        await q.edit_message_text(f"❌ Market data unavailable for {asset}")
        return

    df = Indicators.enrich(df)
    plan = SignalEngine.generate(df)

    if not plan:
        await q.edit_message_text(f"⚠️ No high-probability setup for {asset}")
        return

    msg = (
        f"🧠 Trade Plan\n\n"
        f"Asset: {asset}\n"
        f"Direction: {plan['direction']}\n"
        f"Entry: {round(plan['price'], 6)}\n"
        f"Stop Loss: {round(plan['sl'], 6)}\n"
        f"Take Profit: {round(plan['tp'], 6)}\n"
        f"Confidence: {round(plan['confidence'] * 100, 1)}%"
    )

    await q.edit_message_text(msg)

    TradeMonitor.open_trades[asset] = plan
    asyncio.create_task(
        TradeMonitor.watch(asset, q.from_user.id, context)
    )

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()