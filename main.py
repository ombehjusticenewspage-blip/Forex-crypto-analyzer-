import asyncio
import requests
import pandas as pd
import ta
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
TWELVEDATA_KEY = "ca1acbf0cedb4488b130c59252891c5e"
ALPHAVANTAGE_KEY = "EOGVA134GOOP2UMU"

MIN_CANDLES = 50
MONITOR_INTERVAL = 300
COOLDOWN = 3600
TOP_N = 3
USER_AGENT = "crypto-analyzer/3.0"

CRYPTOS = {
    "BTCUSDT":"BTC/USD","ETHUSDT":"ETH/USD","BNBUSDT":"BNB/USD",
    "XRPUSDT":"XRP/USD","SOLUSDT":"SOL/USD","ADAUSDT":"ADA/USD",
    "DOGEUSDT":"DOGE/USD","AVAXUSDT":"AVAX/USD","DOTUSDT":"DOT/USD",
    "MATICUSDT":"MATIC/USD","LTCUSDT":"LTC/USD","LINKUSDT":"LINK/USD",
    "TRXUSDT":"TRX/USD","ATOMUSDT":"ATOM/USD","UNIUSDT":"UNI/USD"
}

TradeMonitorOpen = {}
LastSignalTime = {}

def fetch_twelvedata(symbol, interval):
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol":CRYPTOS[symbol],"interval":interval,"outputsize":500,"apikey":TWELVEDATA_KEY},
            headers={"User-Agent":USER_AGENT}, timeout=12
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
            params={"function":"DIGITAL_CURRENCY_INTRADAY","symbol":base,"market":"USD","apikey":ALPHAVANTAGE_KEY},
            headers={"User-Agent":USER_AGENT}, timeout=12
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
    def analyze(symbol):
        timeframes = ["5min","15min","1h"]
        sl_mult, tp_mult = 1.5, 3
        best_signal = None
        best_score = 0
        for tf in timeframes:
            df = fetch_twelvedata(symbol, tf)
            if df.empty or len(df) < 50:
                continue
            df = Indicators.enrich(df)
            if df.empty:
                continue
            last = df.iloc[-1]
            price = float(last["c"])
            atr = float(last["ATR"])
            score_buy = sum([
                last["EMA50"] > last["EMA200"],
                last["RSI"] > 55,
                last["MACD"] > 0,
                last["ADX"] > 20,
                last["c"] > last["VWAP"],
                last["ST"] > 50,
                last["c"] > last["ICHIMOKU"],
                last["VOL_SPIKE"]
            ])
            score_sell = 8 - score_buy
            if score_buy >= 6 and score_buy > best_score:
                best_signal = ("BUY", price, price - atr*sl_mult, price + atr*tp_mult, score_buy/8)
                best_score = score_buy
            elif score_sell >= 6 and score_sell > best_score:
                best_signal = ("SELL", price, price + atr*sl_mult, price - atr*tp_mult, score_sell/8)
                best_score = score_sell
        return best_signal

async def live_monitor(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    now = datetime.utcnow()
    signals = []
    for asset in CRYPTOS:
        last_time = LastSignalTime.get(asset, datetime.min)
        if (now - last_time).total_seconds() < COOLDOWN:
            continue
        if asset in TradeMonitorOpen:
            continue
        result = MultiTimeframeEngine.analyze(asset)
        if result:
            direction, price, sl, tp, conf = result
            signals.append((conf, asset, direction, price, sl, tp))
    signals.sort(reverse=True)
    for signal in signals[:TOP_N]:
        conf, asset, direction, price, sl, tp = signal
        TradeMonitorOpen[asset] = {"direction":direction,"sl":sl,"tp":tp}
        LastSignalTime[asset] = now
        msg = f"🚀 New Signal\n{asset}: {direction}\nEntry: {price:.2f} SL: {sl:.2f} TP: {tp:.2f} Confidence: {int(conf*100)}%"
        await context.bot.send_message(chat_id, msg)
    remove_assets = []
    for asset, plan in TradeMonitorOpen.items():
        df = fetch_twelvedata(asset,"5min")
        if df.empty:
            continue
        price = float(df.iloc[-1]["c"])
        if plan["direction"]=="BUY" and (price<=plan["sl"] or price>=plan["tp"]):
            await context.bot.send_message(chat_id,f"{asset} closed at {price}")
            remove_assets.append(asset)
        elif plan["direction"]=="SELL" and (price>=plan["sl"] or price<=plan["tp"]):
            await context.bot.send_message(chat_id,f"{asset} closed at {price}")
            remove_assets.append(asset)
    for a in remove_assets:
        TradeMonitorOpen.pop(a,None)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📡 Strong crypto signals activated, top 3 per interval...")
    context.job_queue.run_repeating(live_monitor, interval=MONITOR_INTERVAL, first=1, chat_id=update.effective_chat.id)

if __name__=="__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.run_polling()