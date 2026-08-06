"""
Streamlit Cloud entrypoint.

Cloud is configured to run streamlit_app.py and the main file path cannot be
changed after deploy, so this file just hands off to the real app. Importing
dashboard runs it top to bottom, which is exactly what Streamlit does with any
script — there is nothing else to call.

Everything lives in dashboard.py: the Setups / All tickers / Options /
Memecoins / Backtest tabs, the timeframe profiles and the scan engine.
Edit that file, not this one.
"""

import dashboard  # noqa: F401  (import side effects ARE the app)
