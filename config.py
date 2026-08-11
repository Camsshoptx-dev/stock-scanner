"""
config.py

Central configuration for the stock scanner project.
Store non-secret settings here. For secrets (email password, API keys),
use environment variables loaded via a .env file (see email_alerts.py).
"""

# List of tickers the scanner will check.
# Replace with your own watchlist, or load dynamically (e.g. S&P 500 list).
WATCHLIST = ("SPY,QQQ,IWM,DIA,SMH,XLF,XLE,XLK,TQQQ,SOXL,"
             "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AVGO,ORCL,NFLX,"
             "CRM,ADBE,AMD,MU,INTC,TSM,QCOM,ARM,MRVL,AMAT,"
             "LRCX,PLTR,NOW,SNOW,CRWD,PANW,DDOG,NET,APP,IONQ,"
             "UBER,ABNB,SHOP,DASH,RBLX,SPOT,SNAP,"
             "JPM,BAC,WFC,GS,MS,C,SCHW,V,MA,BRK-B,"
             "COIN,HOOD,PYPL,SOFI,AFRM,MSTR,MARA,RIOT,"
             "LLY,UNH,JNJ,PFE,MRK,ABBV,MRNA,"
             "XOM,CVX,COP,SLB,OXY,MPC,"
             "WMT,COST,HD,TGT,NKE,SBUX,MCD,DIS,"
             "CAT,BA,GE,UPS,LMT,DAL,"
             "RIVN,LCID,PLUG,SOUN,RKLB,ACHR,CLSK,SMCI,"
             "ES=F,MES=F,NQ=F,MNQ=F,YM=F,RTY=F,"
             "CL=F,MCL=F,GC=F,MGC=F,SI=F,HG=F,NG=F,ZB=F,ZN=F,6E=F").split(",")

# How much historical data to pull for indicator calculations.
# Valid yfinance periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
HISTORY_PERIOD = "6mo"
HISTORY_INTERVAL = "1d"  # 1m, 5m, 15m, 1h, 1d, etc.

# Indicator parameters
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
EMA_WINDOW = 20
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD parameters (standard defaults: 12/26/9)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands parameters
BB_WINDOW = 20
BB_NUM_STD = 2.0

# Scanner behavior
# Example: only alert on RSI oversold/overbought crosses
SCAN_INTERVAL_SECONDS = 300  # how often to run the scanner in a loop (5 min)

# How many symbols per yfinance request. Lower this if you start seeing
# rate-limit errors; raise it for fewer, larger requests.
DOWNLOAD_CHUNK_SIZE = 20

# Email alert settings (non-secret). Actual credentials go in a .env file:
#   EMAIL_ADDRESS=you@gmail.com
#   EMAIL_APP_PASSWORD=your_gmail_app_password
#   ALERT_RECIPIENT=you@gmail.com
EMAIL_SUBJECT_PREFIX = "[Stock Scanner Alert]"

# Backtesting settings
BACKTEST_INITIAL_CAPITAL = 10_000
BACKTEST_PERIOD = "2y"
