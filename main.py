import os, time, asyncio, requests, numpy as np, pandas as pd, ta
from textblob import TextBlob
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes


from binance.client import Client

BINANCE_API_KEY = "DmIS0QxaE6XsUnVCgpewCV4KdFuxFjmQhfhm2YRIiyPQsRQ6vcl1QTp1tWpxLN3Z"        

def fetch_binance_ohlcv(symbol='BTCUSDT', interval='5m', limit=200):
    try:
        klines = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time','o','h','l','c','v','close_time',
            'quote_asset_volume','number_of_trades',
            'taker_buy_base','taker_buy_quote','ignore'
        ])
        df['time'] = pd.to_datetime(df['open_time'], unit='ms')
        df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
        
        return df[['time','o','h','l','c','v']]
    except Exception as e:
        print("[Binance fetch error]", e)
        return pd.DataFrame()

TELEGRAM_TOKEN ="8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
NEWS_API_KEY = "ca1acbf0cedb4488b130c59252891c5e"
MONITOR_INTERVAL = 90  

CRYPTOS = {"BTCUSDT":"bitcoin", "ETHUSDT":"ethereum", "BNBUSDT":"binancecoin", "XRPUSDT":"ripple", "SOLUSDT":"solana"}
FOREX = {"EURUSD":"EURUSD=X", "GBPUSD":"GBPUSD=X", "USDJPY":"USDJPY=X", "USDCHF":"USDCHF=X", "XAUUSD":"XAUUSD=X"}
ALL_ASSETS = dict(**CRYPTOS, **FOREX)

class DataSource:
    @staticmethod
    def fetch_crypto(symbol):
        
        df = fetch_binance_ohlcv(symbol, interval='5m', limit=200)
        if not df.empty:
            print(f"[Binance] Fetched {len(df)} candles for {symbol}")
            df['o'] = df['o'].shift(1)  
            df = df.dropna().reset_index(drop=True)
            return df
        print(f"[Binance] Could not fetch {symbol}; falling back to CoinGecko")
        
        if symbol not in CRYPTOS:
            print(f"[fetch_crypto] Symbol {symbol} not in CRYPTOS")
            return pd.DataFrame()
        coin_id = CRYPTOS[symbol]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency":"usd","days":2,"interval":"minute"}
        try:
            r = requests.get(url, params=params, timeout=15)
            print(f"[Crypto] GET {url}, status={r.status_code}")
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                print(f"[Crypto][{symbol}] API error: {data['error']}")
                return pd.DataFrame()
            prices = data.get("prices",[])
            if len(prices)<50:
                print(f"[Crypto][{symbol}] Insufficient prices: {len(prices)}")
                return pd.DataFrame()
            df = pd.DataFrame(prices, columns=["time","c"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["v"] = np.nan  
            df["o"] = df["c"].shift(1)
            df["h"] = df[["o","c"]].max(axis=1)
            df["l"] = df[["o","c"]].min(axis=1)
            df = df.dropna().reset_index(drop=True)
            print(f"[CoinGecko] {symbol} OHLCV shape: {df.shape}")
            return df
        except Exception as e:
            print(f"[CoinGecko fetch error] {e}")
            return pd.DataFrame()
    @staticmethod
    def fetch_forex(symbol):
        if symbol not in FOREX:
            print(f"[fetch_forex] Symbol {symbol} not in FOREX")
            return pd.DataFrame()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{FOREX[symbol]}"
        params = {"interval":"5m","range":"1d"}
        try:
            r = requests.get(url, params=params, timeout=10)
            print(f"[Forex] GET {url}, status={r.status_code}")
            r.raise_for_status()
            j = r.json()
            if 'chart' not in j or 'result' not in j['chart'] or not j['chart']['result']:
                print(f"[Forex][{symbol}] Malformed Yahoo Finance data")
                return pd.DataFrame()
            res = j['chart']['result'][0]
            t = res.get('timestamp', [])
            prices = res['indicators']['quote'][0].get('close', [])
            highs  = res['indicators']['quote'][0].get('high', [])
            lows   = res['indicators']['quote'][0].get('low', [])
            if not t or not prices or len(t) != len(prices):
                print(f"[Forex][{symbol}] Malformed timing/prices length {len(t)} {len(prices)}")
                return pd.DataFrame()
            df = pd.DataFrame({
                "time": pd.to_datetime(t, unit="s"),
                "c": prices,
                "h": highs,
                "l": lows
            })
            df["o"] = df["c"].shift(1)
            df = df.dropna().reset_index(drop=True)
            print(f"[Forex] {symbol} OHLCV shape: {df.shape}")
            return df
        except Exception as e:
            print(f"[Forex fetch error] {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch(symbol):
        if symbol in CRYPTOS: return DataSource.fetch_crypto(symbol)
        elif symbol in FOREX:  return DataSource.fetch_forex(symbol)
        else:
            print(f"[fetch] Unknown symbol: {symbol}")
            return pd.DataFrame()

class NewsSentiment:
    @staticmethod
    def score(symbol):
        name = symbol.replace("USDT","").replace("=","")
        url = "https://newsapi.org/v2/everything"
        params = {"q":name,"language":"en","sortBy":"publishedAt","apiKey":NEWS_API_KEY,"pageSize":5}
        try:
            r = requests.get(url, params=params, timeout=7).json()
            articles = r.get("articles",[])
            if not articles:
                print(f"[NewsSentiment] No articles found for {name}")
                return 0
            score = sum(TextBlob(a["title"]).sentiment.polarity for a in articles)/len(articles)
            print(f"[NewsSentiment] {name}: {score:.3f}")
            return score
        except Exception as e:
            print(f"[News sentiment error] {e}")
            return 0

class Indicators:
    @staticmethod
    def enrich(df):
        if df.empty:
            print("[Indicators] Input DF is empty")
            return pd.DataFrame()
        try:
            df["RSI"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
            df["MACD"] = ta.trend.MACD(df["c"]).macd_diff()
            df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range()
            df["EMA20"] = ta.trend.EMAIndicator(df["c"],20).ema_indicator()
            bb_up = ta.volatility.BollingerBands(df["c"],20).bollinger_hband()
            bb_low = ta.volatility.BollingerBands(df["c"],20).bollinger_lband()
            df["BB"] = bb_up - bb_low
            df = df.dropna().reset_index(drop=True)
            print(f"[Indicators] Output shape: {df.shape}")
            return df
        except Exception as e:
            print(f"[Indicator error] {e}")
            return pd.DataFrame()

class SignalEngine:
    @staticmethod
    def generate(df, sentiment):
        if df.empty or df.shape[0] < 2:
            print("[SignalEngine] Insufficient indicator-enriched data")
            return dict(direction="HOLD",confidence=0,price=0,sl=0,tp=0,atr=0)
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
        print(f"[SignalEngine] Decision: {direction}, Conf={confidence:.2f}, Price={price}")
        return dict(direction=direction,confidence=confidence,price=price,sl=sl,tp=tp,atr=atr)

class TradeMonitor:
    open_trades = {}
    @classmethod
    async def watch(cls, asset, user_id, context):
        if asset not in cls.open_trades: return
        plan = cls.open_trades[asset]
        entry, sl, tp, direction = plan["price"], plan["sl"], plan["tp"], plan["direction"]
        print(f"[TradeMonitor] Watching {asset} for user {user_id} ({direction})")
        while True:
            await asyncio.sleep(MONITOR_INTERVAL)
            df = DataSource.fetch(asset)
            if df.empty: continue
            price = df.iloc[-1]["c"]
            print(f"[Monitor] {asset} Price: {price} - SL:{sl} TP:{tp}")
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
    if df.empty:
        await q.edit_message_text("❌ Market data unavailable - see logs for details."); return
    df = Indicators.enrich(df)
    if df.empty:
        await q.edit_message_text("❌ Market data enrichment failed. See logs."); return
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