import aiohttp, asyncio, pandas as pd, ta, datetime, numpy as np
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
TWELVEDATA_KEY = "ca1acbf0cedb4488b130c59252891c5e"
ALPHAVANTAGE_KEY = "EOGVA134GOOP2UMU"
NEWS_API_KEY = "qDGIzb9o2OttTxWNvBLMDyZD9KbdQ0qaPHvupsjH"

MIN_CANDLES = 150
MIN_ATR_PCT = 0.3
MIN_BACKTEST_WR = 55
SCAN_INTERVAL = 180
TRACK_INTERVAL = 60
NEWS_LIMIT = 10
MIN_VOLUME_USD = 500000

CRYPTOS = {
    "BTCUSDT":"BTC/USD","ETHUSDT":"ETH/USD","BNBUSDT":"BNB/USD",
    "XRPUSDT":"XRP/USD","SOLUSDT":"SOL/USD","ADAUSDT":"ADA/USD",
    "DOGEUSDT":"DOGE/USD","AVAXUSDT":"AVAX/USD","DOTUSDT":"DOT/USD",
    "MATICUSDT":"MATIC/USD","LTCUSDT":"LTC/USD","LINKUSDT":"LINK/USD",
    "TRXUSDT":"TRX/USD","ATOMUSDT":"ATOM/USD","UNIUSDT":"UNI/USD"
}

active_trades = {}
stats = {"wins":0,"losses":0,"be":0}
ml_scaler = StandardScaler()
ml_model = SGDClassifier(max_iter=1000, tol=1e-3)

def session_ok():
    h = datetime.datetime.utcnow().hour
    return 7 <= h <= 20

async def fetch(session, symbol, interval="15min", outputsize=1000):
    try:
        async with session.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol":CRYPTOS[symbol],"interval":interval,"outputsize":outputsize,"apikey":TWELVEDATA_KEY},
            timeout=10
        ) as r:
            j = await r.json()
            if "values" in j:
                rows=[{"c":float(v["close"]),"h":float(v["high"]),"l":float(v["low"]),"v":float(v.get("volume",0))} for v in reversed(j["values"])]
                return pd.DataFrame(rows)
    except:
        pass
    try:
        async with session.get(
            "https://www.alphavantage.co/query",
            params={"function":"DIGITAL_CURRENCY_INTRADAY","symbol":symbol.replace("USDT",""),"market":"USD","apikey":ALPHAVANTAGE_KEY},
            timeout=10
        ) as r:
            j = await r.json()
            if "Time Series (Digital Currency Intraday)" in j:
                rows=[{"c":float(v["4a. close (USD)"]),"h":float(v["2. high (USD)"]),"l":float(v["3. low (USD)"]),"v":float(v["5. volume"])} for k,v in sorted(j["Time Series (Digital Currency Intraday)"].items())]
                return pd.DataFrame(rows)
    except:
        return pd.DataFrame()
    return pd.DataFrame()

def enrich(df):
    if len(df)<MIN_CANDLES: return pd.DataFrame()
    df["RSI"]=ta.momentum.RSIIndicator(df["c"],14).rsi()
    df["EMA50"]=ta.trend.EMAIndicator(df["c"],50).ema_indicator()
    df["EMA200"]=ta.trend.EMAIndicator(df["c"],200).ema_indicator()
    df["MACD"]=ta.trend.MACD(df["c"]).macd_diff()
    df["ADX"]=ta.trend.ADXIndicator(df["h"],df["l"],df["c"]).adx()
    df["ATR"]=ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"]).average_true_range()
    df["BOLL_H"]=ta.volatility.BollingerBands(df["c"]).bollinger_hband()
    df["BOLL_L"]=ta.volatility.BollingerBands(df["c"]).bollinger_lband()
    df["STOCH_K"]=ta.momentum.StochasticOscillator(df["h"],df["l"],df["c"]).stoch()
    df["CCI"]=ta.trend.CCIIndicator(df["h"],df["l"],df["c"]).cci()
    df["MFI"]=ta.volume.MFIIndicator(df["h"],df["l"],df["c"],df["v"]).money_flow_index()
    df["OBV"]=ta.volume.OnBalanceVolumeIndicator(df["c"],df["v"]).on_balance_volume()
    df["VOLUSD"]=df["v"]*df["c"]
    return df.dropna()

def backtest_score(df, window=200):
    wins=0; losses=0
    start=max(MIN_CANDLES, len(df)-window)
    for i in range(start,len(df)-3):
        r=df.iloc[i]; s=0
        s+=2 if r["EMA50"]>r["EMA200"] else -2
        s+=1 if r["RSI"]>55 else -1 if r["RSI"]<45 else 0
        s+=1 if r["MACD"]>0 else -1
        s+=1 if r["ADX"]>20 else 0
        if s>=3: wins+=1 if df["c"].iloc[i+3]>r["c"] else losses+1
        if s<=-3: wins+=1 if df["c"].iloc[i+3]<r["c"] else losses+1
    t=wins+losses
    return (wins/t)*100 if t else 0

async def fetch_news(session,symbol,limit=NEWS_LIMIT):
    try:
        query=symbol.replace("USDT","")+" crypto"
        async with session.get(
            "https://newsapi.org/v2/everything",
            params={"q":query,"pageSize":limit,"sortBy":"publishedAt","apiKey":NEWS_API_KEY},
            timeout=10
        ) as r:
            j = await r.json()
            if "articles" not in j: return []
            return [{"title":a["title"],"description":a.get("description",""),"url":a["url"]} for a in j["articles"]]
    except:
        return []

def sentiment_score(news):
    if not news: return 0
    scores=[]
    for n in news:
        t_score=TextBlob(n["title"]).sentiment.polarity
        d_score=TextBlob(n["description"]).sentiment.polarity if n["description"] else 0
        scores.append((t_score*0.7+d_score*0.3))
    return np.mean(scores)

def online_train(df, news_sent=0):
    if len(df)<MIN_CANDLES+5: return
    X=[]
    y=[]
    for i in range(MIN_CANDLES,len(df)-3):
        r=df.iloc[i]
        features=[r["RSI"], r["EMA50"], r["EMA200"], r["MACD"], r["ADX"], r["ATR"],
                  r["BOLL_H"], r["BOLL_L"], r["STOCH_K"], r["CCI"], r["MFI"], r["OBV"],
                  r["VOLUSD"], news_sent]
        X.append(features)
        next_close = df["c"].iloc[i+3]
        y.append(1 if next_close>r["c"] else -1)
    if X:
        X_scaled = ml_scaler.fit_transform(X)
        ml_model.partial_fit(X_scaled, y, classes=[-1,1])

def signal(df, news_sentiment=0, weight=1):
    last=df.iloc[-1]
    if last["VOLUSD"]<MIN_VOLUME_USD: return None
    atr_pct=(last["ATR"]/last["c"])*100
    if atr_pct<MIN_ATR_PCT: return None
    if backtest_score(df)<MIN_BACKTEST_WR: return None
    s=0
    s+=2.5 if last["EMA50"]>last["EMA200"] else -2.5
    s+=1.2 if last["RSI"]>55 else -1.2 if last["RSI"]<45 else 0
    s+=1.5 if last["MACD"]>0 else -1.5
    s+=1.2 if last["ADX"]>20 else 0
    s+=1 if last["STOCH_K"]>80 else -1 if last["STOCH_K"]<20 else 0
    s+=1 if last["CCI"]>100 else -1 if last["CCI"]<-100 else 0
    s+=0.8 if last["BOLL_H"]<last["c"]<last["BOLL_L"] else 0
    s+=1 if last["MFI"]>80 else -1 if last["MFI"]<20 else 0
    s+=1 if last["OBV"]>0 else -1
    s+=1.5 if news_sentiment>0.05 else -1.5 if news_sentiment<-0.05 else 0
    s*=weight
    features=np.array([last["RSI"], last["EMA50"], last["EMA200"], last["MACD"], last["ADX"], last["ATR"],
                       last["BOLL_H"], last["BOLL_L"], last["STOCH_K"], last["CCI"], last["MFI"], last["OBV"],
                       last["VOLUSD"], news_sentiment]).reshape(1,-1)
    features_scaled=ml_scaler.fit_transform(features)
    pred=ml_model.predict(features_scaled) if hasattr(ml_model,"coef_") else 0
    s+=pred[0] if pred is not None else 0
    if s>=3: d="BUY"
    elif s<=-3: d="SELL"
    else: return None
    p=last["c"]; atr=last["ATR"]
    sl_offset = atr*1.5*np.clip(1+atr_pct/10,1,2)
    tp_offset = atr*3*np.clip(1+atr_pct/10,1,2)
    return {"dir":d,"entry":p,"sl":p-sl_offset if d=="BUY" else p+sl_offset,
            "tp":p+tp_offset if d=="BUY" else p-tp_offset,"atr":atr,"score":s,"be":False}

async def btc_trend(session):
    df=enrich(await fetch(session,"BTCUSDT"))
    if df.empty: return None
    r=df.iloc[-1]
    return "BUY" if r["EMA50"]>r["EMA200"] else "SELL"

async def multi_tf_signal(session,symbol):
    if not session_ok(): return None
    btc_dir=await btc_trend(session)
    timeframes=[("1min",0.5),("5min",1),("15min",1.5),("1h",2),("5h",3),("1d",4)]
    dfs=[]
    for tf,_ in timeframes: dfs.append(enrich(await fetch(session,symbol,tf)))
    if any(df.empty for df in dfs): return None
    news=await fetch_news(session,symbol)
    news_sent=sentiment_score(news)
    for df,(tf,weight) in zip(dfs,timeframes): online_train(df, news_sent)
    sigs=[]
    dirs=[]
    for (df,(tf,weight)) in zip(dfs,timeframes):
        s=signal(df,news_sent,weight)
        if s is None: return None
        sigs.append(s)
        dirs.append(s["dir"])
    if len(set(dirs))>1: return None
    if symbol!="BTCUSDT" and dirs[0]!=btc_dir: return None
    return sigs[0]

async def scan(context):
    signals=[]
    async with aiohttp.ClientSession() as s:
        for sym in CRYPTOS:
            if sym in active_trades: continue
            sig=await multi_tf_signal(s,sym)
            if sig: signals.append((sym,sig))
    signals.sort(key=lambda x: abs(x[1]["score"]),reverse=True)
    top3=signals[:3]
    if top3:
        msg="🚀 TOP CRYPTO SIGNALS\n\n"
        async with aiohttp.ClientSession() as s:
            for i,(sym,s) in enumerate(top3,1):
                news=await fetch_news(s,sym)
                msg+=f"{i}. {sym}\nDir: {s['dir']}\nEntry: {round(s['entry'],5)}\nSL: {round(s['sl'],5)}\nTP: {round(s['tp'],5)}\nScore: {s['score']}\n"
                if news:
                    msg+="📰 News:\n"
                    for n in news: msg+=f"{n['title']}\n{n['url']}\n"
                msg+="\n"
        await context.bot.send_message(chat_id=context.job.chat_id,text=msg)
        for sym,s in top3: active_trades[sym]=s

async def track(context):
    async with aiohttp.ClientSession() as s:
        for sym in list(active_trades):
            df=await fetch(s,sym)
            if df.empty: continue
            price=df["c"].iloc[-1]
            t=active_trades[sym]
            if not t["be"]:
                if t["dir"]=="BUY" and price>=t["entry"]+t["atr"]: t["sl"]=t["entry"]; t["be"]=True
                if t["dir"]=="SELL" and price<=t["entry"]-t["atr"]: t["sl"]=t["entry"]; t["be"]=True
            tp=price>=t["tp"] if t["dir"]=="BUY" else price<=t["tp"]
            sl=price<=t["sl"] if t["dir"]=="BUY" else price>=t["sl"]
            if tp or sl:
                if tp: stats["wins"]+=1
                elif t["be"]: stats["be"]+=1
                else: stats["losses"]+=1
                total=stats["wins"]+stats["losses"]
                wr=round((stats["wins"]/total)*100,2) if total else 0
                await context.bot.send_message(chat_id=context.job.chat_id,
                    text=f"{sym} {'TP' if tp else 'SL'} @ {round(price,5)}\nWins {stats['wins']} Loss {stats['losses']} BE {stats['be']}\nWinRate {wr}%")
                del active_trades[sym]

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    kb=[[InlineKeyboardButton(k,callback_data=k)] for k in CRYPTOS]
    await update.message.reply_text("📡 Click crypto to analyze or wait for top signals:",reply_markup=InlineKeyboardMarkup(kb))
    context.job_queue.run_repeating(scan,SCAN_INTERVAL,chat_id=update.effective_chat.id)
    context.job_queue.run_repeating(track,TRACK_INTERVAL,chat_id=update.effective_chat.id)

async def analyze_callback(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    await q.answer()
    sym=q.data
    async with aiohttp.ClientSession() as s:
        sig=await multi_tf_signal(s,sym)
        if not sig:
            await q.edit_message_text(f"⚠️ No clear signal for {sym} right now")
            return
        active_trades[sym]=sig
        news=await fetch_news(s,sym)
        msg=f"{sym} Analysis\nDir: {sig['dir']}\nEntry: {round(sig['entry'],5)}\nSL: {round(sig['sl'],5)}\nTP: {round(sig['tp'],5)}\nScore: {sig['score']}\n"
        if news:
            msg+="📰 News:\n"
            for n in news: msg+=f"{n['title']}\n{n['url']}\n"
        await q.edit_message_text(msg)

if __name__=="__main__":
    app=ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(analyze_callback))
    app.run_polling()