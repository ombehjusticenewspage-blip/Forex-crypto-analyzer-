import asyncio
import aiohttp
import pandas as pd
import ta
from datetime import datetime
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

async def fetch_twelvedata(session, symbol, interval):
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": CRYPTOS[symbol],
            "interval": interval,
            "outputsize": 500,
            "apikey": TWELVEDATA_KEY
        }
        async with session.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=12) as r:
            j = await r.json()
            if "values" not in j or not j["values"]:
                return await fetch_alphavantage(session, symbol)
            rows = [{
                "time": pd.to_datetime(v["datetime"]),
                "o": float(v["open"]),
                "h": float(v["high"]),
                "l": float(v["low"]),
                "c": float(v["close"]),
                "v": float(v.get("volume", 0))
            } for v in reversed(j["values"])]
            return pd.DataFrame(rows)
    except:
        return await fetch_alphavantage(session, symbol)

async def fetch_alphavantage(session, symbol):
    try:
        base = symbol.replace("USDT","")
        url = "https://www.alphavantage.co/query"
        params = {
            "function":"DIGITAL_CURRENCY_INTRADAY",
            "symbol": base,
            "market":"USD",
            "apikey": ALPHAVANTAGE_KEY
        }
        async with session.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=12) as r:
            j = await r.json()
            ts_key = next((x for x in j if "Time Series" in x), None)
            if not ts_key or not j[ts_key]:
                return pd.DataFrame()
            rows = [{
                "time": pd.to_datetime(t),
                "o": float(v.get("1a. open (USD)",0)),
                "h": float(v.get("2. high (USD)",0)),
                "l": float(v.get("3. low (USD)",0)),
                "c": float(v.get("4a. close (USD)",0)),
                "v": 0
            } for t,v in j[ts_key].items()]
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
    async def analyze(symbol):
        async with aiohttp.ClientSession() as session:
            df = await fetch_twelvedata(session, symbol, "5min")
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
                direction = "BUY"
            elif bearish:
                direction = "SELL"
            else:
                return None
            price = float(last["c"])
            atr = float(last["ATR"])
            sl = price - atr*1.6 if direction=="BUY" else price + atr*1.6
            tp = price + atr*3.2 if direction=="BUY" else price - atr*3.2
            return direction, price, sl, tp, 0.92

class TradeMonitor:
    open_trades = {}
    @staticmethod
    async def watch(asset,user_id,context):
        while asset in TradeMonitor.open_trades:
            await asyncio.sleep(MONITOR_INTERVAL)
            async with aiohttp.ClientSession() as session:
                df = await fetch_twelvedata(session, asset, "5min")
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
    kb = [[InlineKeyboardButton(k,callback_data=k)] for k in CRYPTOS]
    await update.message.reply_text("Select Asset:",reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    await q.edit_message_text(f"Analyzing {asset}...")
    result = await MultiTimeframeEngine.analyze(asset)
    if not result:
        await q.edit_message_text(f"No high-probability setup for {asset} or data unavailable")
        return
    direction, price, sl, tp, conf = result
    await q.edit_message_text(
        f"🚀 Trade Signal\n\nAsset: {asset}\nDirection: {direction}\nEntry: {price}\nSL: {sl}\nTP: {tp}\nConfidence: {int(conf*100)}%"
    )
    TradeMonitor.open_trades[asset] = {"direction":direction,"sl":sl,"tp":tp}
    asyncio.create_task(TradeMonitor.watch(asset,q.from_user.id,context))

if __name__=="__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()