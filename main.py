import os
import asyncio
import requests
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ta
from textblob import TextBlob
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
try:
    from binance.client import Client as BinanceClient
except Exception:
    BinanceClient = None
try:
    import ccxt
except Exception:
    ccxt = None

TELEGRAM_TOKEN = "8543111323:AAHcBtUS7dZsBl2bG74HhmPPyIoectRw8xo"
NEWS_API_KEY = pub_d2023a3a059649e5bc3fc9ca4efa5a5e
MONITOR_INTERVAL = "90"
MIN_CANDLES = "30"
USER_AGENT = "forex-crypto-analyzer/1.0"

CRYPTOS = {"BTCUSDT":"bitcoin","ETHUSDT":"ethereum","BNBUSDT":"binancecoin","XRPUSDT":"ripple","SOLUSDT":"solana"}
FOREX = {"EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","USDCHF":"USDCHF=X","XAUUSD":"XAUUSD=X"}
ALL_ASSETS = dict(**CRYPTOS, **FOREX)

binance_client = None
if BINANCE_API_KEY and BINANCE_API_SECRET and BinanceClient is not None:
    try:
        binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
    except Exception:
        binance_client = None

def normalize_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    cols = df.columns
    if "time" not in cols and "timestamp" in cols:
        df = df.rename(columns={"timestamp":"time"})
    if "time" in df.columns:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
    required = ["time","o","h","l","c","v"]
    for r in required:
        if r not in df.columns:
            if r == "v":
                df[r] = np.nan
            else:
                df[r] = df.get("c", np.nan)
    return df[required]

def fetch_binance_public(symbol="BTCUSDT", interval="5m", limit=500, diagnostics=None):
    try:
        url = "https://api.binance.com/api/v3/klines"
        r = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit}, timeout=10, headers={"User-Agent":USER_AGENT})
        if diagnostics is not None:
            diagnostics.append(f"Binance public HTTP {r.status_code}")
        r.raise_for_status()
        klines = r.json()
        df = pd.DataFrame(klines, columns=[
            'open_time','o','h','l','c','v','close_time',
            'quote_asset_volume','number_of_trades',
            'taker_buy_base','taker_buy_quote','ignore'
        ])
        df['time'] = pd.to_datetime(df['open_time'], unit='ms')
        df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
        return normalize_df(df)
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"Binance public error: {e}")
        return pd.DataFrame()

def fetch_ccxt_exchanges(symbol, diagnostics=None):
    if ccxt is None:
        if diagnostics is not None:
            diagnostics.append("ccxt not installed")
        return pd.DataFrame()
    exchanges = ["binance","kraken","coinbasepro","bitstamp","bitfinex"]
    for ex in exchanges:
        try:
            exchange = getattr(ccxt, ex)()
            if hasattr(exchange, "fetch_ohlcv"):
                ohlcv = exchange.fetch_ohlcv(symbol.replace("USDT","/USDT") if "USDT" in symbol else symbol, timeframe="5m", limit=500)
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=["time","o","h","l","c","v"])
                    df["time"] = pd.to_datetime(df["time"], unit="ms")
                    if diagnostics is not None:
                        diagnostics.append(f"ccxt {ex} OK {len(df)}")
                    return normalize_df(df)
        except Exception as e:
            if diagnostics is not None:
                diagnostics.append(f"ccxt {ex} error: {e}")
            continue
    return pd.DataFrame()

def fetch_coingecko(symbol, diagnostics=None):
    try:
        coin = CRYPTOS.get(symbol)
        if not coin:
            if diagnostics is not None:
                diagnostics.append("CoinGecko unknown coin")
            return pd.DataFrame()
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        for attempt in [{"days":2,"interval":"minute"},{"days":7,"interval":"hourly"},{"days":30,"interval":"daily"}]:
            r = requests.get(url, params={"vs_currency":"usd","days":attempt["days"],"interval":attempt["interval"]}, timeout=12, headers={"User-Agent":USER_AGENT})
            if diagnostics is not None:
                diagnostics.append(f"CoinGecko HTTP {r.status_code} days={attempt['days']}")
            if r.status_code != 200:
                continue
            data = r.json()
            prices = data.get("prices",[])
            if len(prices) < 5:
                continue
            df = pd.DataFrame(prices, columns=["time","c"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["v"] = np.nan
            df["o"] = df["c"].shift(1)
            df["h"] = df[["o","c"]].max(axis=1)
            df["l"] = df[["o","c"]].min(axis=1)
            df = df.dropna().reset_index(drop=True)
            return normalize_df(df)
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"CoinGecko error: {e}")
    return pd.DataFrame()

def fetch_alphavantage_crypto(symbol, diagnostics=None):
    if not ALPHAVANTAGE_KEY:
        if diagnostics is not None:
            diagnostics.append("AlphaVantage key not configured")
        return pd.DataFrame()
    try:
        base = symbol[:-4]
        url = "https://www.alphavantage.co/query"
        r = requests.get(url, params={"function":"DIGITAL_CURRENCY_INTRADAY","symbol":base,"market":"USD","apikey":ALPHAVANTAGE_KEY}, timeout=12, headers={"User-Agent":USER_AGENT})
        if diagnostics is not None:
            diagnostics.append(f"AlphaVantage crypto HTTP {r.status_code}")
        if r.status_code != 200:
            return pd.DataFrame()
        j = r.json()
        timekey = next((k for k in j.keys() if "Time Series" in k), None)
        if not timekey:
            return pd.DataFrame()
        ts = j[timekey]
        rows = []
        for t, vals in ts.items():
            rows.append({"time":pd.to_datetime(t),"o":float(vals.get("1a. open (USD)", vals.get("1b. open (USD)", vals.get("1. open",0)))),"h":float(vals.get("2. high (USD)", vals.get("2. high",0))),"l":float(vals.get("3. low (USD)", vals.get("3. low",0))),"c":float(vals.get("4a. close (USD)", vals.get("4b. close (USD)", vals.get("4. close",0)))),"v":np.nan})
        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        return normalize_df(df)
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"AlphaVantage crypto error: {e}")
        return pd.DataFrame()

def fetch_twelvedata(symbol, diagnostics=None):
    if not TWELVEDATA_KEY:
        if diagnostics is not None:
            diagnostics.append("TwelveData key not configured")
        return pd.DataFrame()
    try:
        sd = symbol if "/" in symbol else (symbol[:-4] + "/" + symbol[-4:]) if "USDT" in symbol else symbol
        url = "https://api.twelvedata.com/time_series"
        r = requests.get(url, params={"symbol":sd,"interval":"5min","outputsize":500,"apikey":TWELVEDATA_KEY}, timeout=12, headers={"User-Agent":USER_AGENT})
        if diagnostics is not None:
            diagnostics.append(f"TwelveData HTTP {r.status_code}")
        j = r.json()
        if "values" not in j:
            return pd.DataFrame()
        rows = []
        for v in reversed(j["values"]):
            rows.append({"time":pd.to_datetime(v["datetime"]),"o":float(v["open"]),"h":float(v["high"]),"l":float(v["low"]),"c":float(v["close"]),"v":float(v.get("volume",np.nan))})
        df = pd.DataFrame(rows)
        return normalize_df(df)
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"TwelveData error: {e}")
        return pd.DataFrame()

def fetch_finnhub_crypto(symbol, diagnostics=None):
    if not FINNHUB_KEY:
        if diagnostics is not None:
            diagnostics.append("Finnhub key not configured")
        return pd.DataFrame()
    try:
        base = symbol[:-4]
        url = "https://finnhub.io/api/v1/crypto/candle"
        r = requests.get(url, params={"symbol":f"BINANCE:{base}USDT","resolution":"5","count":500,"token":FINNHUB_KEY}, timeout=12, headers={"User-Agent":USER_AGENT})
        if diagnostics is not None:
            diagnostics.append(f"Finnhub HTTP {r.status_code}")
        j = r.json()
        if j.get("s") != "ok":
            return pd.DataFrame()
        times = [datetime.utcfromtimestamp(t) for t in j.get("t",[])]
        df = pd.DataFrame({"time":times,"o":j.get("o",[]),"h":j.get("h",[]),"l":j.get("l",[]),"c":j.get("c",[]),"v":j.get("v",[])})
        return normalize_df(df)
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"Finnhub error: {e}")
        return pd.DataFrame()

def fetch_exchangerate_timeseries(base, quote, days=60, diagnostics=None):
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        url = "https://api.exchangerate.host/timeseries"
        r = requests.get(url, params={"start_date":start.isoformat(),"end_date":end.isoformat(),"base":base,"symbols":quote}, timeout=12, headers={"User-Agent":USER_AGENT})
        if diagnostics is not None:
            diagnostics.append(f"exchangerate.host HTTP {r.status_code}")
        r.raise_for_status()
        j = r.json()
        rates = j.get("rates",{})
        rows = []
        for d, vals in sorted(rates.items()):
            rate = vals.get(quote)
            if rate is None:
                continue
            rows.append({"time":pd.to_datetime(d),"c":float(rate)})
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"exchangerate.host error: {e}")
        return pd.DataFrame()

def synthesize_ohlcv_from_close(close_df, needed=MIN_CANDLES):
    if close_df.empty:
        return pd.DataFrame()
    closes = close_df["c"].tolist()
    if len(closes) >= needed:
        chosen = closes[-needed:]
    else:
        chosen = closes + [closes[-1]] * (needed - len(closes))
    times = [datetime.utcnow() - timedelta(minutes=5*(needed-1-i)) for i in range(needed)]
    df = pd.DataFrame({"time":times,"c":chosen})
    df["o"] = df["c"].shift(1).fillna(df["c"])
    df["h"] = df["c"] * 1.0005
    df["l"] = df["c"] * 0.9995
    df["v"] = np.nan
    return normalize_df(df)

def fetch_yahoo_forex(symbol, diagnostics=None):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{FOREX[symbol]}"
        r = requests.get(url, params={"interval":"5m","range":"1d"}, timeout=12, headers={"User-Agent":USER_AGENT})
        if diagnostics is not None:
            diagnostics.append(f"Yahoo HTTP {r.status_code}")
        r.raise_for_status()
        j = r.json()
        res = j.get("chart",{}).get("result",[])
        if not res:
            return pd.DataFrame()
        res = res[0]
        ts = res.get("timestamp",[])
        quotes = res.get("indicators",{}).get("quote",[])
        if not quotes:
            return pd.DataFrame()
        c = quotes[0].get("close",[])
        h = quotes[0].get("high",[])
        l = quotes[0].get("low",[])
        rows = []
        for i,t in enumerate(ts):
            rows.append({"time":pd.to_datetime(t,unit="s"),"o":None,"h":h[i],"l":l[i],"c":c[i],"v":np.nan})
        df = pd.DataFrame(rows)
        df["o"] = df["c"].shift(1).fillna(df["c"])
        return normalize_df(df)
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"Yahoo error: {e}")
        return pd.DataFrame()

class DataSource:
    def fetch_crypto(symbol):
        diagnostics = []
        try:
            df = fetch_binance_public(symbol, diagnostics=diagnostics)
            if not df.empty() if hasattr(df,"empty") else not df.empty:
                if df.shape[0] >= MIN_CANDLES:
                    return df, diagnostics
        except Exception:
            pass
        try:
            df_ccxt = fetch_ccxt_exchanges(symbol, diagnostics=diagnostics)
            if not df_ccxt.empty and df_ccxt.shape[0] >= MIN_CANDLES:
                return df_ccxt, diagnostics
        except Exception:
            pass
        try:
            df_cg = fetch_coingecko(symbol, diagnostics=diagnostics)
            if not df_cg.empty and df_cg.shape[0] >= MIN_CANDLES:
                return df_cg, diagnostics
        except Exception:
            pass
        try:
            df_av = fetch_alphavantage_crypto(symbol, diagnostics=diagnostics)
            if not df_av.empty and df_av.shape[0] >= MIN_CANDLES:
                return df_av, diagnostics
        except Exception:
            pass
        try:
            df_td = fetch_twelvedata(symbol, diagnostics=diagnostics)
            if not df_td.empty and df_td.shape[0] >= MIN_CANDLES:
                return df_td, diagnostics
        except Exception:
            pass
        try:
            df_fh = fetch_finnhub_crypto(symbol, diagnostics=diagnostics)
            if not df_fh.empty and df_fh.shape[0] >= MIN_CANDLES:
                return df_fh, diagnostics
        except Exception:
            pass
        try:
            price = None
            url = "https://api.coingecko.com/api/v3/simple/price"
            coin = CRYPTOS.get(symbol)
            if coin:
                r = requests.get(url, params={"ids":coin,"vs_currencies":"usd"}, timeout=8, headers={"User-Agent":USER_AGENT})
                j = r.json()
                price = j.get(coin,{}).get("usd")
            if price:
                df_syn = synthesize_ohlcv_from_close(pd.DataFrame([{"time":datetime.utcnow(),"c":price}]), needed=MIN_CANDLES)
                diagnostics.append("Synthesized crypto series from CoinGecko simple price")
                return df_syn, diagnostics
        except Exception as e:
            diagnostics.append(f"synthesis error: {e}")
        return pd.DataFrame(), diagnostics

    def fetch_forex(symbol):
        diagnostics = []
        if symbol not in FOREX:
            diagnostics.append("Unknown forex symbol")
            return pd.DataFrame(), diagnostics
        try:
            df_av = pd.DataFrame()
            if ALPHAVANTAGE_KEY:
                try:
                    url = "https://www.alphavantage.co/query"
                    params = {"function":"FX_INTRADAY","from_symbol":symbol[:3],"to_symbol":symbol[3:],"interval":"5min","outputsize":"full","apikey":ALPHAVANTAGE_KEY}
                    r = requests.get(url, params=params, timeout=12, headers={"User-Agent":USER_AGENT})
                    diagnostics.append(f"AlphaVantage HTTP {r.status_code}")
                    j = r.json()
                    key = next((k for k in j.keys() if "Time Series" in k), None)
                    if key:
                        ts = j[key]
                        rows = []
                        for t, vals in ts.items():
                            rows.append({"time":pd.to_datetime(t),"o":float(vals.get("1. open",0)),"h":float(vals.get("2. high",0)),"l":float(vals.get("3. low",0)),"c":float(vals.get("4. close",0)),"v":np.nan})
                        df_av = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
                except Exception as e:
                    diagnostics.append(f"AlphaVantage error: {e}")
            if not df_av.empty and df_av.shape[0] >= MIN_CANDLES:
                return normalize_df(df_av), diagnostics
        except Exception:
            pass
        try:
            df_td = pd.DataFrame()
            if TWELVEDATA_KEY:
                try:
                    sym = symbol[:3] + "/" + symbol[3:]
                    url = "https://api.twelvedata.com/time_series"
                    r = requests.get(url, params={"symbol":sym,"interval":"5min","outputsize":500,"apikey":TWELVEDATA_KEY}, timeout=12, headers={"User-Agent":USER_AGENT})
                    diagnostics.append(f"TwelveData HTTP {r.status_code}")
                    j = r.json()
                    if "values" in j:
                        rows = []
                        for v in reversed(j["values"]):
                            rows.append({"time":pd.to_datetime(v["datetime"]),"o":float(v["open"]),"h":float(v["high"]),"l":float(v["low"]),"c":float(v["close"]),"v":float(v.get("volume",np.nan))})
                        df_td = pd.DataFrame(rows)
                except Exception as e:
                    diagnostics.append(f"TwelveData error: {e}")
            if not df_td.empty and df_td.shape[0] >= MIN_CANDLES:
                return normalize_df(df_td), diagnostics
        except Exception:
            pass
        try:
            df_y = fetch_yahoo_forex(symbol, diagnostics=diagnostics)
            if not df_y.empty and df_y.shape[0] >= MIN_CANDLES:
                return df_y, diagnostics
        except Exception:
            pass
        try:
            close_df = fetch_exchangerate_timeseries(symbol[:3], symbol[3:], diagnostics=diagnostics)
            if not close_df.empty:
                df_syn = synthesize_ohlcv_from_close(close_df, needed=max(MIN_CANDLES, len(close_df)))
                diagnostics.append("Synthesized forex series from exchangerate.host")
                return df_syn, diagnostics
        except Exception as e:
            diagnostics.append(f"exchangerate synthesis error: {e}")
        return pd.DataFrame(), diagnostics

    def diagnose(symbol):
        parts = []
        if symbol in CRYPTOS:
            try:
                r = requests.get("https://api.binance.com/api/v3/ping", timeout=6, headers={"User-Agent":USER_AGENT})
                parts.append(f"Binance ping HTTP {r.status_code}")
            except Exception as e:
                parts.append(f"Binance ping error: {e}")
            if ccxt is not None:
                parts.append("ccxt available")
            else:
                parts.append("ccxt not available")
            try:
                r2 = requests.get(f"https://api.coingecko.com/api/v3/coins/{CRYPTOS[symbol]}/market_chart", params={"vs_currency":"usd","days":2}, timeout=8, headers={"User-Agent":USER_AGENT})
                parts.append(f"CoinGecko HTTP {r2.status_code}")
            except Exception as e:
                parts.append(f"CoinGecko error: {e}")
            if ALPHAVANTAGE_KEY:
                parts.append("AlphaVantage configured")
            else:
                parts.append("AlphaVantage not configured")
            if TWELVEDATA_KEY:
                parts.append("TwelveData configured")
            else:
                parts.append("TwelveData not configured")
            if FINNHUB_KEY:
                parts.append("Finnhub configured")
            else:
                parts.append("Finnhub not configured")
        elif symbol in FOREX:
            try:
                r = requests.get("https://api.exchangerate.host/latest", timeout=6, headers={"User-Agent":USER_AGENT})
                parts.append(f"exchangerate.host HTTP {r.status_code}")
            except Exception as e:
                parts.append(f"exchangerate.host error: {e}")
            try:
                r2 = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{FOREX[symbol]}", timeout=8, headers={"User-Agent":USER_AGENT})
                parts.append(f"Yahoo HTTP {r2.status_code}")
            except Exception as e:
                parts.append(f"Yahoo error: {e}")
            if ALPHAVANTAGE_KEY:
                parts.append("AlphaVantage configured")
            if TWELVEDATA_KEY:
                parts.append("TwelveData configured")
        else:
            parts.append("Unknown symbol")
        if NEWS_API_KEY:
            try:
                r = requests.get("https://newsapi.org/v2/everything", params={"q":"bitcoin","pageSize":1,"apiKey":NEWS_API_KEY}, timeout=6, headers={"User-Agent":USER_AGENT})
                parts.append(f"NewsAPI HTTP {r.status_code}")
            except Exception as e:
                parts.append(f"NewsAPI error: {e}")
        else:
            parts.append("NewsAPI not configured")
        return "\n".join(parts)

class NewsSentiment:
    def score(symbol):
        if not NEWS_API_KEY:
            return 0.0
        q = symbol.replace("USDT","").replace("=","")
        try:
            r = requests.get("https://newsapi.org/v2/everything", params={"q":q,"pageSize":5,"apiKey":NEWS_API_KEY}, timeout=8, headers={"User-Agent":USER_AGENT})
            r.raise_for_status()
            j = r.json()
            arts = j.get("articles",[])
            if not arts:
                return 0.0
            return float(sum(TextBlob(a.get("title","")).sentiment.polarity for a in arts)/max(1,len(arts)))
        except Exception:
            return 0.0

class Indicators:
    def enrich(df):
        if df is None or df.empty:
            return pd.DataFrame()
        if df.shape[0] < MIN_CANDLES:
            return pd.DataFrame()
        dfc = df.copy().reset_index(drop=True)
        try:
            dfc["RSI"] = ta.momentum.RSIIndicator(dfc["c"],14).rsi()
            dfc["MACD"] = ta.trend.MACD(dfc["c"]).macd_diff()
            dfc["ATR"] = ta.volatility.AverageTrueRange(dfc["h"],dfc["l"],dfc["c"],14).average_true_range()
            dfc["EMA20"] = ta.trend.EMAIndicator(dfc["c"],20).ema_indicator()
            dfc = dfc.dropna().reset_index(drop=True)
            return dfc
        except Exception:
            return pd.DataFrame()

class SignalEngine:
    def generate(df, sentiment):
        if df.empty or df.shape[0] < 2:
            return dict(direction="HOLD",confidence=0.0,price=0.0,sl=0.0,tp=0.0,atr=0.0)
        last = df.iloc[-1]
        signals = []
        if last["RSI"] < 35 and last["MACD"] > 0:
            signals.append("BUY")
        elif last["RSI"] > 70 and last["MACD"] < 0:
            signals.append("SELL")
        if last["c"] > last["EMA20"]:
            signals.append("BUY")
        elif last["c"] < last["EMA20"]:
            signals.append("SELL")
        s_bias = "BUY" if sentiment > 0.05 else "SELL" if sentiment < -0.05 else ""
        if s_bias:
            signals.append(s_bias)
        direction = "BUY" if signals.count("BUY") > signals.count("SELL") else "SELL" if signals.count("SELL") > signals.count("BUY") else "HOLD"
        confidence = min(max(abs(last["RSI"]-50)/50 + abs(sentiment),0),1)
        atr = float(last.get("ATR",0) or 0)
        price = float(last["c"])
        sl = price - atr if direction=="BUY" else price + atr
        tp = price + atr*2 if direction=="BUY" else price - atr*2
        return dict(direction=direction,confidence=confidence,price=price,sl=sl,tp=tp,atr=atr)

class TradeMonitor:
    open_trades = {}
    async def watch(cls, asset, user_id, context):
        if asset not in cls.open_trades:
            return
        plan = cls.open_trades[asset]
        entry, sl, tp, direction = plan["price"], plan["sl"], plan["tp"], plan["direction"]
        while True:
            await asyncio.sleep(MONITOR_INTERVAL)
            df, _ = DataSource.fetch(asset) if hasattr(DataSource.fetch,"__call__") else (pd.DataFrame(),[])
            if df.empty:
                continue
            price = float(df.iloc[-1]["c"])
            if direction == "BUY" and price < sl:
                await context.bot.send_message(user_id, f"❗ {asset}: Price fell below stop-loss. Last Price: {round(price,6)}")
                break
            if direction == "SELL" and price > sl:
                await context.bot.send_message(user_id, f"❗ {asset}: Price rose above stop-loss. Last Price: {round(price,6)}")
                break
            if direction == "BUY" and price > tp:
                await context.bot.send_message(user_id, f"✅ {asset}: Take-profit hit. Last Price: {round(price,6)}")
                break
            if direction == "SELL" and price < tp:
                await context.bot.send_message(user_id, f"✅ {asset}: Take-profit hit. Last Price: {round(price,6)}")
                break

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(k, callback_data=k)] for k in ALL_ASSETS]
    await update.message.reply_text("Select Asset (Crypto/Forex):", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    await q.edit_message_text(f"Analyzing {asset} ...")
    if asset in CRYPTOS:
        df, diagnostics = DataSource.fetch_crypto(asset)
    else:
        df, diagnostics = DataSource.fetch_forex(asset)
    if df.empty:
        diag_text = "\n".join(diagnostics) if isinstance(diagnostics, list) else str(diagnostics)
        reason = DataSource.diagnose(asset)
        await q.edit_message_text(f"❌ Market data unavailable for {asset}.\n\nFetch diagnostics:\n{diag_text}\n\nProbe diagnostics:\n{reason}")
        return
    df_ind = Indicators.enrich(df)
    if df_ind.empty:
        await q.edit_message_text(f"❌ Not enough market data ({df.shape[0]} candles).")
        return
    sentiment = NewsSentiment.score(asset)
    plan = SignalEngine.generate(df_ind, sentiment)
    msg = (f"🧠 Trade Plan\n\nAsset: {asset}\nDirection: {plan['direction']}\nEntry: {round(plan['price'],6)}\nSL: {round(plan['sl'],6)}\nTP: {round(plan['tp'],6)}\nConfidence: {round(plan['confidence']*100,1)}%\n(sentiment={round(sentiment,3)})")
    await q.edit_message_text(msg)
    user_id = q.from_user.id
    if plan["direction"] in ("BUY","SELL"):
        TradeMonitor.open_trades[asset] = {**plan,"user_id":user_id}
        asyncio.create_task(TradeMonitor.watch(asset,user_id,context))

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Running diagnostics...")
    reports = []
    for s in list(ALL_ASSETS):
        try:
            reports.append(f"{s}:\n{DataSource.diagnose(s)}")
        except Exception as e:
            reports.append(f"{s}:\nerror: {e}")
    out = "\n\n".join(reports)
    if len(out) > 3800:
        out = out[:3800] + "\n\n...[truncated]"
    await update.message.reply_text(out)

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise SystemExit(1)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("diag", diag_cmd))
    app.add_handler(CallbackQueryHandler(analyze))
    app.run_polling()