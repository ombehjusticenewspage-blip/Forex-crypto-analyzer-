import asyncio
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
TWELVEDATA_KEY = "ca1acbf0cedb4488b130c59252891c5e"
ALPHAVANTAGE_KEY = "EOGVA134GOOP2UMU"

MIN_CANDLES = 150
MONITOR_INTERVAL = 90
USER_AGENT = "crypto-analyzer/3.0"

CRYPTOS = {
    "BTCUSDT":"BTC/USD","ETHUSDT":"ETH/USD","BNBUSDT":"BNB/USD",
    "XRPUSDT":"XRP/USD","SOLUSDT":"SOL/USD","ADAUSDT":"ADA/USD",
    "DOGEUSDT":"DOGE/USD","AVAXUSDT":"AVAX/USD","DOTUSDT":"DOT/USD",
    "MATICUSDT":"MATIC/USD","LTCUSDT":"LTC/USD","LINKUSDT":"LINK/USD",
    "TRXUSDT":"TRX/USD","ATOMUSDT":"ATOM/USD","UNIUSDT":"UNI/USD"
}

USER_MODE = {}

def fetch_twelvedata(symbol, interval):
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol":CRYPTOS[symbol],
                "interval":interval,
                "outputsize":500,
                "apikey":TWELVEDATA_KEY
            },
            headers={"User-Agent":USER_AGENT},
            timeout=12
        )
        j = r.json()
        if "values" not in j or not j["values"]:
            return fetch_alphavantage(symbol)
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
        return fetch_alphavantage(symbol)

def fetch_alphavantage(symbol):
    try:
        base = symbol.replace("USDT","")
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function":"DIGITAL_CURRENCY_INTRADAY",
                "symbol":base,
                "market":"USD",
                "apikey":ALPHAVANTAGE_KEY
            },
            headers={"User-Agent":USER_AGENT},
            timeout=12
        )
        j = r.json()
        ts_key = next((x for x in j if "Time Series" in x), None)
        if not ts_key or not j[ts_key]:
            return pd.DataFrame()
        rows = []
        for t,v in j[ts_key].items():
            rows.append({
                "time": pd.to_datetime(t),
                "o": float(v.get("1a. open (USD)",0)),
                "h": float(v.get("2. high (USD)",0)),
                "l": float(v.get("3. low (USD)",0)),
                "c": float(v.get("4a. close (USD)",0)),
                "v": 0
            })
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    except:
        return pd.DataFrame()

class Indicators:
    def enrich(df):
        if df.empty or len(df) < 50:
            return pd.DataFrame()
        df = df.copy().reset_index(drop=True)
        df["RSI"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
        macd = ta.trend.MACD(df["c"])
        df["MACD"] = macd.macd_diff()
        df["EMA50"] = ta.trend.EMAIndicator(df["c"],50).ema_indicator()
        df["EMA200"] = ta.trend.EMAIndicator(df["c"],200).ema_indicator()
        df["ADX"] = ta.trend.ADXIndicator(df["h"],df["l"],df["c"]).adx()
        df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"]).average_true_range()
        df["VWAP"] = (df["v"] * (df["h"] + df["l"] + df["c"]) / 3).cumsum() / df["v"].cumsum()
        df["ST"] = ta.trend.STCIndicator(df["c"]).stc()
        df["ICHIMOKU"] = ta.trend.IchimokuIndicator(df["h"],df["l"]).ichimoku_base_line()
        df["VOL_MA"] = df["v"].rolling(20).mean()
        df["VOL_SPIKE"] = df["v"] > df["VOL_MA"] * 1.5
        return df.dropna()

class MultiTimeframeEngine:
    def analyze(symbol, mode):
        if mode == "SCALP":
            timeframes = ["5min","15min"]
            sl_mult, tp_mult = 1.2, 2.2
        else:
            timeframes = ["15min","1h"]
            sl_mult, tp_mult = 1.6, 3.2
        directions = []
        price = atr = None
        for tf in timeframes:
            df = fetch_twelvedata(symbol, tf)
            if df.empty or df.shape[0] < MIN_CANDLES:
                return None
            df = Indicators.enrich(df)
            if df.empty:
                return None
            last = df.iloc[-1]
            bullish = (
                last["EMA50"] > last["EMA200"] and
                last["RSI"] > 55 and
                last["MACD"] > 0 and
                last["ADX"] > 20 and
                last["c"] > last["VWAP"] and
                last["ST"] > 50 and
                last["c"] > last["ICHIMOKU"] and
                last["VOL_SPIKE"]
            )
            bearish = (
                last["EMA50"] < last["EMA200"] and
                last["RSI"] < 45 and
                last["MACD"] < 0 and
                last["ADX"] > 20 and
                last["c"] < last["VWAP"] and
                last["ST"] < 50 and
                last["c"] < last["ICHIMOKU"] and
                last["VOL_SPIKE"]
            )
            if bullish:
                directions.append("BUY")
            elif bearish:
                directions.append("SELL")
            else:
                return None
            price = float(last["c"])
            atr = float(last["ATR"])
        if not all(d == directions[0] for d in directions):
            return None
        direction = directions[0]
        sl = price - atr*sl_mult if direction=="BUY" else price + atr*sl_mult
        tp = price + atr*tp_mult if direction=="BUY" else price - atr*tp_mult
        return direction, price, sl, tp, 0.92

class Backtester:
    def run(symbol, mode):
        df = fetch_twelvedata(symbol,"5min")
        if df.empty or df.shape[0] < MIN_CANDLES:
            return None
        df = Indicators.enrich(df)
        if df.empty:
            return None
        wins = losses = 0
        for i in range(200, len(df)):
            slice_df = df.iloc[:i]
            result = MultiTimeframeEngine.analyze(symbol, mode)
            if not result:
                continue
            direction, entry, sl, tp, _ = result
            price = df.iloc[i]["c"]
            if direction == "BUY":
                wins += 1 if price >= tp else 0
                losses += 1 if price <= sl else 0
            else:
                wins += 1 if price <= tp else 0
                losses += 1 if price >= sl else 0
        total = wins + losses
        winrate = (wins / total * 100) if total > 0 else 0
        return wins, losses, winrate

class TradeMonitor:
    open_trades = {}
    async def watch(asset,user_id,context):
        while asset in TradeMonitor.open_trades:
            await asyncio.sleep(MONITOR_INTERVAL)
            df = fetch_twelvedata(asset,"5min")
            if df.empty:
                continue
            price = float(df.iloc[-1]["c"])
            plan = TradeMonitor.open_trades[asset]
            if plan["direction"]=="BUY" and (price<=plan["sl"] or price>=plan["tp"]):
                await context.bot.send_message(user_id,f"{asset} closed at {price}")
                TradeMonitor.open_trades.pop(asset,None)
            if plan["direction"]=="SELL" and (price>=plan["sl"] or price<=plan["tp"]):
                await context.bot.send_message(user_id,f"{asset} closed at {price}")
                TradeMonitor.open_trades.pop(asset,None)

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("⚡ Scalping",callback_data="MODE_SCALP")],
        [InlineKeyboardButton("📈 Swing",callback_data="MODE_SWING")]
    ]
    await update.message.reply_text("Choose trading style:",reply_markup=InlineKeyboardMarkup(kb))

async def mode_select(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    USER_MODE[q.from_user.id] = "SCALP" if "SCALP" in q.data else "SWING"
    kb = [[InlineKeyboardButton(k,callback_data=k)] for k in CRYPTOS]
    await q.edit_message_text("Select Asset:",reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    mode = USER_MODE.get(q.from_user.id,"SWING")
    await q.edit_message_text(f"Analyzing {asset} ({mode})...")
    result = MultiTimeframeEngine.analyze(asset,mode)
    if not result:
        await q.edit_message_text(f"No high-probability setup for {asset} or data unavailable")
        return
    direction,price,sl,tp,conf = result
    await q.edit_message_text(
        f"🚀 Trade Signal\n\nAsset: {asset}\nMode: {mode}\nDirection: {direction}\nEntry: {price}\nSL: {sl}\nTP: {tp}\nConfidence: {int(conf*100)}%"
    )
    TradeMonitor.open_trades[asset] = {"direction":direction,"sl":sl,"tp":tp}
    asyncio.create_task(TradeMonitor.watch(asset,q.from_user.id,context))

async def backtest(update:Update,context:ContextTypes.DEFAULT_TYPE):
    asset = context.args[0] if context.args else "BTCUSDT"
    mode = USER_MODE.get(update.effective_user.id,"SWING")
    result = Backtester.run(asset,mode)
    if not result:
        await update.message.reply_text("Not enough data for backtest")
        return
    wins, losses, winrate = result
    await update.message.reply_text(
        f"📊 Backtest Result\n\nAsset: {asset}\nMode: {mode}\nWins: {wins}\nLosses: {losses}\nWinrate: {round(winrate,2)}%"
    )

if __name__=="__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CommandHandler("backtest",backtest))
    app.add_handler(CallbackQueryHandler(mode_select,pattern="MODE_"))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()