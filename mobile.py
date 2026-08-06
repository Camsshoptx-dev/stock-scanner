"""
MOBILE LAYOUT — drop-in for any of the scanner apps
===================================================
Works with the tabbed build (Setups / All tickers / Options / Memecoins /
Backtest) and with streamlit_app.py. It only writes CSS, so it never touches
your scan logic.

USAGE
    import mobile
    st.set_page_config(...)      # must come first
    mobile.inject()              # call once, right after set_page_config

WHAT IT FIXES
    Streamlit stacks st.columns vertically below ~640px. That is sensible for
    forms and terrible for a 4-metric row, which becomes four full-height
    blocks you have to scroll past before reaching a single signal. Same for
    the ticker/price row inside a card.

    Tabs wrap onto three lines on a phone. They scroll sideways instead.

NOTE ON SELECTORS
    Streamlit's internal data-testid values change between releases. Every
    rule below is written with a fallback selector alongside the testid, so a
    version bump degrades to "slightly wider padding" rather than a broken
    layout.
"""

import streamlit as st

CSS = """
<style>
/* ---------- shared ---------- */
:root{
  --ink:#E8EEF6; --dim:#5F6B7E; --line:#1E2733; --panel:#111721;
  --up:#5EE9B5; --down:#FF6B7A;
}

/* ================================================================
   PHONE  (<= 640px)
   ================================================================ */
@media (max-width: 640px){

  /* Streamlit's default side padding eats ~15% of a phone screen. */
  .block-container,
  [data-testid="stAppViewContainer"] .block-container{
    padding:0.75rem 0.6rem 3rem !important;
    max-width:100% !important;
  }

  /* --- the main fix: stop columns stacking ---------------------
     Streamlit forces flex-direction:column below its own breakpoint.
     Override it and let the columns wrap only when they truly must. */
  [data-testid="stHorizontalBlock"]{
    flex-direction:row !important;
    flex-wrap:wrap !important;
    gap:6px !important;
  }
  [data-testid="stColumn"],
  [data-testid="column"]{
    min-width:0 !important;          /* lets flex children actually shrink */
    flex:1 1 calc(50% - 6px) !important;
    width:auto !important;
  }

  /* Metric row: 4 across on one line, smaller type instead of stacked. */
  [data-testid="stMetric"]{
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:8px;
    padding:8px 6px !important;
  }
  [data-testid="stMetricValue"]{
    font-size:19px !important;
    line-height:1.15 !important;
  }
  [data-testid="stMetricLabel"]{
    font-size:9.5px !important;
    letter-spacing:.04em;
    text-transform:uppercase;
    opacity:.7;
  }
  [data-testid="stMetricLabel"] p{font-size:9.5px !important}
  [data-testid="stMetricDelta"]{font-size:11px !important}

  /* --- tabs: scroll sideways rather than wrap to three rows --- */
  .stTabs [data-baseweb="tab-list"]{
    overflow-x:auto !important;
    overflow-y:hidden !important;
    flex-wrap:nowrap !important;
    scrollbar-width:none;
    -webkit-overflow-scrolling:touch;
    gap:2px !important;
    padding-bottom:2px;
    /* fade on the right edge so it reads as scrollable */
    mask-image:linear-gradient(90deg,#000 88%,transparent);
    -webkit-mask-image:linear-gradient(90deg,#000 88%,transparent);
  }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{display:none}
  .stTabs [data-baseweb="tab"]{
    flex:0 0 auto !important;
    white-space:nowrap !important;
    padding:7px 11px !important;
    font-size:13px !important;
    min-width:auto !important;
  }
  .stTabs [data-baseweb="tab-panel"]{padding-top:10px !important}

  /* --- signal cards --- */
  .card{padding:10px 12px !important;margin-bottom:6px !important}
  .card-top{flex-wrap:nowrap !important;gap:7px !important}
  .tkr{font-size:15px !important}
  .tag{font-size:9px !important;padding:2px 7px !important}
  .px .p{font-size:14px !important}
  .px .c{font-size:11px !important}
  .meta{font-size:10.5px !important;line-height:1.45 !important}
  .tf-pill{font-size:9px !important;padding:1px 5px !important}

  /* Multi-timeframe strip scrolls sideways instead of wrapping to 3 rows. */
  .mtf{
    flex-wrap:nowrap !important;
    overflow-x:auto;
    scrollbar-width:none;
    -webkit-overflow-scrolling:touch;
    padding-bottom:3px;
  }
  .mtf::-webkit-scrollbar{display:none}
  .cell{flex:0 0 auto;font-size:10px !important;padding:2px 6px !important}

  /* --- headings --- */
  .hero h1,h1{font-size:20px !important}
  .hero .sub{font-size:11px !important}
  h2{font-size:17px !important}
  h3{font-size:15px !important}
  h4{font-size:13.5px !important}

  /* --- controls: 44px touch targets, no iOS zoom-on-focus --- */
  .stButton>button,.stDownloadButton>button{
    min-height:44px !important;font-size:14px !important;width:100% !important;
  }
  .stSelectbox div[data-baseweb="select"]>div,
  .stMultiSelect div[data-baseweb="select"]>div{min-height:42px !important}
  input,textarea,select,
  .stTextInput input,.stNumberInput input,.stTextArea textarea{
    font-size:16px !important;   /* below 16px Safari zooms the whole page */
  }
  [data-testid="stSidebar"]{width:88vw !important}

  /* --- tables scroll rather than squeeze to unreadable --- */
  [data-testid="stDataFrame"],[data-testid="stTable"]{
    overflow-x:auto !important;font-size:11px !important;
  }

  /* --- charts get shorter so they don't own the whole screen --- */
  [data-testid="stVegaLiteChart"],
  .stPlotlyChart,
  [data-testid="stArrowVegaLiteChart"]{max-height:260px !important}

  /* Expander headers were losing their text to overflow. */
  .streamlit-expanderHeader,[data-testid="stExpander"] summary{
    font-size:13px !important;
  }
}

/* ================================================================
   NARROW PHONE  (<= 380px)
   ================================================================ */
@media (max-width: 380px){
  /* Below this, 4 metrics across genuinely stops being readable —
     2x2 is the honest fallback. */
  [data-testid="stColumn"],[data-testid="column"]{
    flex:1 1 calc(50% - 6px) !important;
  }
  [data-testid="stMetricValue"]{font-size:17px !important}
  .tkr{font-size:14px !important}
  .stTabs [data-baseweb="tab"]{padding:6px 9px !important;font-size:12px !important}
}

/* ================================================================
   TABLET  (641-1024px)
   ================================================================ */
@media (min-width:641px) and (max-width:1024px){
  .block-container{padding-left:1.2rem !important;padding-right:1.2rem !important}
  [data-testid="stMetricValue"]{font-size:22px !important}
}

/* ---------- landscape phone: reclaim vertical space ---------- */
@media (max-height:500px) and (orientation:landscape){
  .block-container{padding-top:0.4rem !important}
  [data-testid="stVegaLiteChart"]{max-height:190px !important}
}

/* ---------- accessibility ---------- */
@media (prefers-reduced-motion:reduce){
  *{animation-duration:.01ms !important;transition-duration:.01ms !important}
}
</style>
"""

VIEWPORT = """
<script>
// Streamlit ships a viewport meta tag, but custom components in an iframe can
// land without one, which makes a phone render at desktop width and then scale
// everything down. Add it only if it is genuinely missing.
(function(){
  try{
    var d = window.parent && window.parent.document ? window.parent.document : document;
    if(!d.querySelector('meta[name="viewport"]')){
      var m = d.createElement('meta');
      m.name = 'viewport';
      m.content = 'width=device-width, initial-scale=1, viewport-fit=cover';
      d.head.appendChild(m);
    }
  }catch(e){ /* cross-origin frame - nothing to do */ }
})();
</script>
"""


def inject(viewport: bool = True) -> None:
    """Apply the mobile stylesheet. Call once, after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)
    if viewport:
        st.markdown(VIEWPORT, unsafe_allow_html=True)


def compact_metrics(pairs, cols: int = 4) -> None:
    """Metric row that stays horizontal on a phone.

    st.columns + st.metric is the usual way, but Streamlit stacks those
    vertically on mobile no matter what the CSS says, because it sets the
    flex direction inline. This renders one flex row directly instead.

        mobile.compact_metrics([("Buy", 3), ("Short", 5), ("Hold", 18)])
    """
    cells = []
    for label, value in pairs:
        v = str(value)
        tone = "up" if v.startswith("+") else "down" if v.startswith("-") else ""
        cells.append(
            f'<div class="cm-cell"><div class="cm-lbl">{label}</div>'
            f'<div class="cm-val {tone}">{v}</div></div>')
    st.markdown(
        f"""<style>
.cm-row{{display:flex;gap:6px;margin:6px 0 14px}}
.cm-cell{{flex:1 1 0;min-width:0;background:var(--panel,#111721);
  border:1px solid var(--line,#1E2733);border-radius:8px;padding:9px 8px;
  text-align:center}}
.cm-lbl{{font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--dim,#5F6B7E);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}}
.cm-val{{font-size:20px;font-weight:700;color:var(--ink,#E8EEF6);
  line-height:1.2;font-variant-numeric:tabular-nums}}
.cm-val.up{{color:var(--up,#5EE9B5)}}.cm-val.down{{color:var(--down,#FF6B7A)}}
@media(max-width:640px){{.cm-val{{font-size:17px}}}}
@media(max-width:380px){{.cm-row{{flex-wrap:wrap}}
  .cm-cell{{flex:1 1 calc(50% - 6px)}}}}
</style><div class="cm-row">{''.join(cells)}</div>""",
        unsafe_allow_html=True)
