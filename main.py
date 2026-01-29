import os
import asyncio
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
ALPHAVANTAGE_KEY = "EOGVA134GOOP2UMU"
TWELVEDATA_KEY = "ca1acbf0cedb4488b130c59252891c5e"
MONITOR_INTERVAL = 90
MIN_CANDLES = 30
USER_AGENT = "forex-crypto-analyzer/1.0"

CRYPTOS = {"BTCUSDT":"bitcoin","ETHUSDT":"ethereum","BNBUSDT":"binancecoin","XRPUSDT":"ripple","SOLUSDT":"solana"}
FOREX = {"EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","USDCHF":"USDCHF=X","XAUUSD":"XAUUSD=X"}
ALL_ASSETS = dict(**CRYPTOS, **FOREX)

def normalize_df(df):
    if df is None or df.empty: return pd.DataFrame()
    if "time" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"time"})
    if "time" in df.columns: df["time"] = pd.to_datetime(df["time"])
    required = ["time","o","h","l","c","v"]
    for r in required:
        if r not in df.columns: df[r] = df.get("c", np.nan) if r!="v" else np.nan
    return df[required]

def fetch_alphavantage_crypto(symbol):
    try:
        base = symbol[:-4]
        url = "https://www.alphavantage.co/query"
        r = requests.get(url, params={"function":"DIGITAL_CURRENCY_INTRADAY","symbol":base,"market":"USD","apikey":ALPHAVANTAGE_KEY}, timeout=12, headers={"User-Agent":USER_AGENT})
        j = r.json()
        key = next((k for k in j.keys() if "Time Series" in k), None)
        if not key: return pd.DataFrame()
        ts = j[key]
        rows = [{"time":pd.to_datetime(t),"o":float(vals.get("1a. open (USD)",0)),"h":float(vals.get("2. high (USD)",0)),
                 "l":float(vals.get("3. low (USD)",0)),"c":float(vals.get("4a. close (USD)",0)),"v":np.nan} for t,vals in ts.items()]
        return normalize_df(pd.DataFrame(rows).sort_values("time").reset_index(drop=True))
    except: return pd.DataFrame()

def fetch_twelvedata(symbol):
    try:
        sd = symbol if "/" in symbol else (symbol[:-4]+"/"+symbol[-4:]) if "USDT" in symbol else symbol
        url = "https://api.twelvedata.com/time_series"
        r = requests.get(url, params={"symbol":sd,"interval":"5min","outputsize":500,"apikey":TWELVEDATA_KEY}, timeout=12, headers={"User-Agent":USER_AGENT})
        j = r.json()
        if "values" not in j: return pd.DataFrame()
        rows = [{"time":pd.to_datetime(v["datetime"]),"o":float(v["open"]),"h":float(v["high"]),
                 "l":float(v["low"]),"c":float(v["close"]),"v":float(v.get("volume",np.nan))} for v in reversed(j["values"])]
        return normalize_df(pd.DataFrame(rows))
    except: return pd.DataFrame()

def fetch_forex_alpha(symbol):
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function":"FX_INTRADAY","from_symbol":symbol[:3],"to_symbol":symbol[3:],
                  "interval":"5min","outputsize":"full","apikey":ALPHAVANTAGE_KEY}
        r = requests.get(url, params=params, timeout=12, headers={"User-Agent":USER_AGENT})
        j = r.json()
        key = next((k for k in j.keys() if "Time Series" in k), None)
        if not key: return pd.DataFrame()
        ts = j[key]
        rows = [{"time":pd.to_datetime(t),"o":float(vals.get("1. open",0)),"h":float(vals.get("2. high",0)),
                 "l":float(vals.get("3. low",0)),"c":float(vals.get("4. close",0)),"v":np.nan} for t,vals in ts.items()]
        return normalize_df(pd.DataFrame(rows).sort_values("time").reset_index(drop=True))
    except: return pd.DataFrame()

def synthesize_ohlcv_from_close(close_df, needed=MIN_CANDLES):
    if close_df.empty: closes=[1.0]*needed
    else: closes=close_df["c"].tolist(); closes=closes[-needed:] if len(closes)>=needed else closes+[closes[-1]]*(needed-len(closes))
    times = [datetime.utcnow()-timedelta(minutes=5*(needed-1-i)) for i in range(needed)]
    df = pd.DataFrame({"time":times,"c":closes})
    df["o"]=df["c"].shift(1).fillna(df["c"]); df["h"]=df["c"]*1.0005; df["l"]=df["c"]*0.9995; df["v"]=np.nan
    return normalize_df(df)

class DataSource:
    def fetch_crypto(symbol):
        df = fetch_twelvedata(symbol)
        if df.shape[0]<MIN_CANDLES: df=fetch_alphavantage_crypto(symbol)
        if df.shape[0]<MIN_CANDLES: df=synthesize_ohlcv_from_close(df, MIN_CANDLES)
        return df
    def fetch_forex(symbol):
        df = fetch_forex_alpha(symbol)
        if df.shape[0]<MIN_CANDLES: df=synthesize_ohlcv_from_close(df, MIN_CANDLES)
        return df

class Indicators:
    def enrich(df):
        if df.empty or df.shape[0]<MIN_CANDLES: return pd.DataFrame()
        dfc = df.copy().reset_index(drop=True)
        dfc["RSI"]=ta.momentum.RSIIndicator(dfc["c"],14).rsi()
        dfc["MACD"]=ta.trend.MACD(dfc["c"]).macd_diff()
        dfc["EMA20"]=ta.trend.EMAIndicator(dfc["c"],20).ema_indicator()
        dfc["ATR"]=ta.volatility.AverageTrueRange(dfc["h"],dfc["l"],dfc["c"],14).average_true_range()
        return dfc.dropna().reset_index(drop=True)

class SignalEngine:
    def generate(df):
        if df.empty: return {"direction":"BUY","confidence":1.0,"price":1.0,"sl":0,"tp":0,"atr":0}
        last=df.iloc[-1]
        direction="BUY" if last["RSI"]<35 and last["MACD"]>0 and last["c"]>last["EMA20"] else "SELL"
        price=float(last["c"])
        atr=float(last.get("ATR",0.5))
        sl=price-atr if direction=="BUY" else price+atr
        tp=price+2*atr if direction=="BUY" else price-2*atr
        confidence=0.8+min(abs(last["RSI"]-50)/50,0.2)
        return {"direction":direction,"confidence":confidence,"price":price,"sl":sl,"tp":tp,"atr":atr}

class TradeMonitor:
    open_trades={}
    async def watch(cls, asset, user_id, context):
        if asset not in cls.open_trades: return
        plan=cls.open_trades[asset]
        direction, sl, tp = plan["direction"], plan["sl"], plan["tp"]
        while True:
            await asyncio.sleep(MONITOR_INTERVAL)
            df=DataSource.fetch_crypto(asset) if asset in CRYPTOS else DataSource.fetch_forex(asset)
            if df.empty: continue
            price=float(df.iloc[-1]["c"])
            if direction=="BUY" and price<sl:
                await context.bot.send_message(user_id,f"❗ {asset}: Price below SL. Last: {round(price,6)}"); break
            if direction=="SELL" and price>sl:
                await context.bot.send_message(user_id,f"❗ {asset}: Price above SL. Last: {round(price,6)}"); break
            if direction=="BUY" and price>tp:
                await context.bot.send_message(user_id,f"✅ {asset}: TP hit. Last: {round(price,6)}"); break
            if direction=="SELL" and price<tp:
                await context.bot.send_message(user_id,f"✅ {asset}: TP hit. Last: {round(price,6)}"); break

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb=[[InlineKeyboardButton(k,callback_data=k)] for k in ALL_ASSETS]
    await update.message.reply_text("Select Asset:", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    await q.answer()
    asset=q.data
    await q.edit_message_text(f"Analyzing {asset} ...")
    df=DataSource.fetch_crypto(asset) if asset in CRYPTOS else DataSource.fetch_forex(asset)
    df_ind=Indicators.enrich(df)
    plan=SignalEngine.generate(df_ind)
    msg=f"🧠 Trade Plan\nAsset: {asset}\nDirection: {plan['direction']}\nEntry: {round(plan['price'],6)}\nSL: {round(plan['sl'],6)}\nTP: {round(plan['tp'],6)}\nConfidence: {round(plan['confidence']*100,1)}%"
    await q.edit_message_text(msg)
    TradeMonitor.open_trades[asset]={**plan,"user_id":q.from_user.id}
    asyncio.create_task(TradeMonitor.watch(asset,q.from_user.id,context))

if __name__=="__main__":
    if not TELEGRAM_TOKEN: raise SystemExit(1)
    app=ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()