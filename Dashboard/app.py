"""
ICEBREAKER — ARB Dashboard
Spread Monitor for KC/RC and CC/LCC pairs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ARB Monitor", layout="wide")

DB = Path(__file__).parent.parent / "Database"

KC_FACTOR = 22.0462   # ¢/lbs  →  $/MT

# Contract-pairing convention for the Contract Explorer tab, mirroring
# the CFARB/COCARB month map in Non Fundamental/Seasonality/Code/market_configs.py
# (KC and RC/LRC don't share delivery months, so an anchor month picks a
# matched pair — e.g. anchor "ZX" = KC Z vs RC X same year, "ZF" = KC Z vs RC F next year).
KCRC_MONTH_MAP = {   # anchor -> (KC month, RC month, RC year offset)
    "H":  ("H", "H", 0),
    "K":  ("K", "K", 0),
    "N":  ("N", "N", 0),
    "U":  ("U", "U", 0),
    "ZX": ("Z", "X", 0),
    "ZF": ("Z", "F", 1),
}
CCLCC_MONTH_MAP = {mc: (mc, mc, 0) for mc in ["H", "K", "N", "U", "Z"]}  # CC/LCC share month codes

# ── Palette ───────────────────────────────────────────────────────────────────

PAPER  = "rgba(0,0,0,0)"
PLOT   = "rgba(0,0,0,0)"
GRID   = "rgba(0,0,0,0.06)"
FONT   = "#374151"
MUTED  = "#9ca3af"
NAVY   = "#0a2463"   # selected-pill / current-year colour, same as the Cotton On-Call dashboard
TEAL   = NAVY        # main line colour (name kept - every primary series uses it)
AMBER  = "#c98a1f"   # gold: navy's complement, for the second leg / +-1 sigma bands
RED    = "#c94a4a"   # muted red, same as Cotton On-Call
GREEN  = "#1f9d6f"   # muted green, same as Cotton On-Call

def base_layout(fig, **kw):
    ax = dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
               zerolinecolor=GRID, zerolinewidth=1)
    xo = kw.pop("xaxis", {})
    yo = kw.pop("yaxis", {})
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        font=dict(color=FONT, size=12),
        title_font=dict(color="#111827", size=13),
        margin=dict(t=36, b=20, l=8, r=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT)),
        xaxis={**ax, **xo}, yaxis={**ax, **yo},
        **kw,
    )
    return fig

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_all(mtimes):  # mtimes keys the cache so a new parquet push invalidates it
    gbp = pd.read_parquet(DB / "fx_gbp.parquet")["GBP_USD"]

    # Actual front-month prices (1st/2nd month, no roll adjustment)
    front = {}
    for name in ["KC", "RC", "CC", "LCC"]:
        path = DB / f"front_{name}.parquet"
        front[name] = pd.read_parquet(path) if path.exists() else None

    return gbp, front

_mtimes = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in sorted(DB.glob("*.parquet")))
gbp_raw, front = load_all(_mtimes)
gbp_full = gbp_raw.copy()  # unsliced by date range — Contract Explorer needs historical vintages too

front_available = all(front[n] is not None for n in ["KC", "RC", "CC", "LCC"])

if not front_available:
    st.error("Front-month data not yet ingested — run ingest_front.py first.")
    st.stop()

# ── Per-contract data (for Contract Explorer) ──────────────────────────────────

CONTRACT_FILES = {"KC": "kc_futures.parquet", "RC": "rc_futures.parquet",
                   "CC": "cc_futures.parquet", "LCC": "lcc_futures.parquet"}

@st.cache_data(ttl=3600)
def load_contracts(mtimes):
    data = {}
    for name, fname in CONTRACT_FILES.items():
        path = DB / fname
        data[name] = pd.read_parquet(path) if path.exists() else None
    return data

_contract_mtimes = tuple((DB / f).stat().st_mtime_ns if (DB / f).exists() else 0
                         for f in CONTRACT_FILES.values())
contract_db = load_contracts(_contract_mtimes)
contract_db_available = all(contract_db[n] is not None for n in CONTRACT_FILES)

def continuous_leg(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Continuous single-leg series for one month code: at each date, use
    whichever vintage is nearest its own expiry (smallest LTD). Since an
    older vintage always has a smaller LTD than the next one while it's
    still trading, this picks that vintage for its entire life and only
    switches to the next vintage the day after the current one's last
    trading day — i.e. a stitched, non-overlapping roll for that specific
    contract month, analogous to a front-month series but locked to `month`."""
    sub = df[df["month"] == month].dropna(subset=["LTD", "settlement"])
    if sub.empty:
        return pd.DataFrame(columns=["year", "settlement"])
    idx = sub.groupby("Date")["LTD"].idxmin()
    return sub.loc[idx, ["Date", "year", "settlement"]].sort_values("Date").set_index("Date")

def continuous_pair(db1: pd.DataFrame, m1: str, db2: pd.DataFrame, m2: str, yoff2: int) -> pd.DataFrame:
    """Joins two continuous legs so leg2 always uses the vintage matched to
    leg1's active vintage (year1 + yoff2), not its own independent roll."""
    front1 = continuous_leg(db1, m1)
    if front1.empty:
        return pd.DataFrame(columns=["year1", "leg1", "year2", "leg2"])
    front1 = front1.rename(columns={"year": "year1", "settlement": "leg1"})
    front1["year2"] = front1["year1"] + yoff2

    sub2 = db2[db2["month"] == m2][["Date", "year", "settlement"]].rename(
        columns={"year": "year2", "settlement": "leg2"})

    return front1.reset_index().merge(sub2, on=["Date", "year2"], how="inner").set_index("Date").sort_index()

def get_ltd(df: pd.DataFrame, month: str, year: int):
    """Last trading day for one contract (constant across its rows)."""
    sub = df.loc[(df["month"] == month) & (df["year"] == year), "LTD"].dropna()
    return sub.iloc[0] if not sub.empty else None

def price_asof(df: pd.DataFrame, month: str, year: int, asof: pd.Timestamp):
    """Most recent settlement on or before `asof` for one contract. Returns
    (price, date_used) — date_used lets the UI flag a stale/illiquid quote."""
    sub = df[(df["month"] == month) & (df["year"] == year) & (df["Date"] <= asof)].dropna(subset=["settlement"])
    if sub.empty:
        return None, None
    row = sub.sort_values("Date").iloc[-1]
    return row["settlement"], row["Date"]

def build_term_structure(db1, m_map, anchor_seq, db2, asof: pd.Timestamp, n_maturities: int):
    """All not-yet-expired (anchor, vintage-year) combos across `anchor_seq`,
    sorted by leg1's expiry — i.e. the actual forward sequence of listed
    maturities (H26, K26, ..., ZX26, H27, ...), not one instance per month code."""
    candidates = []
    for yr in range(asof.year - 1, asof.year + 5):
        for a in anchor_seq:
            m1, m2, yoff2 = m_map[a]
            ltd1 = get_ltd(db1, m1, yr)
            if ltd1 is None or ltd1 < asof:
                continue
            candidates.append((ltd1, a, yr, m1, m2, yr + yoff2))
    candidates.sort(key=lambda c: c[0])

    rows = []
    for ltd1, a, yr, m1, m2, y2 in candidates[:n_maturities]:
        p1, d1 = price_asof(db1, m1, yr, asof)
        p2, d2 = price_asof(db2, m2, y2, asof)
        if p1 is None or p2 is None:
            continue
        rows.append(dict(anchor=a, year=yr, m1=m1, m2=m2, y2=y2, ltd1=ltd1,
                          price1=p1, price2=p2, date1=d1, date2=d2))
    return rows

# ── Analytics helpers ─────────────────────────────────────────────────────────

def zscore(spread: pd.Series, window: int) -> pd.Series:
    mu  = spread.rolling(window).mean()
    sig = spread.rolling(window).std()
    return (spread - mu) / sig

def lookback_years(hist_years: list):
    """Past years feeding the seasonal bands/average, from the global Period radio:
    5Y / 10Y = that many most recent completed years, All = every completed year,
    Custom = completed years overlapping the chosen date range (all if none).
    hist_years excludes the current year."""
    if period == "5Y":
        used = hist_years[-5:]
    elif period == "10Y":
        used = hist_years[-10:]
    elif period == "Custom":
        used = [y for y in hist_years if d_start.year <= y <= d_end.year] or hist_years
    else:
        used = hist_years
    if used:
        st.caption(f"Average of {len(used)} yrs ({used[0]}–{used[-1]}), excl. current year.")
    return used

def seasonality_bands_fig(long_df: pd.DataFrame, x_col: str, val_col: str, title: str,
                           xaxis_title: str, yaxis_title: str, current_year: int, last_year: int,
                           reversed_x: bool = False, band_smooth: int = 15, band_years=None):
    """Nested Min-Max / 10-90 / 25-75 percentile bands + dotted average +
    bold current-year line + red last-year line. long_df has one row per
    (x_col, val_col, yr) observation. `band_years` restricts which years feed
    the bands/average (the current/last-year lines always use the full data).

    Daily data across only ~10-15 years means each x value has just that many
    observations (and some fewer, from weekends/holidays), so raw per-day
    percentiles are extremely spiky. The bands (not the actual year lines) are
    smoothed with a centred rolling mean over `band_smooth` neighbouring x values."""
    stat_df = long_df if not band_years else long_df[long_df["yr"].isin(band_years)]
    if stat_df.empty:
        stat_df = long_df
    band = stat_df.groupby(x_col)[val_col].agg(
        lo="min", p10=lambda s: s.quantile(0.10), p25=lambda s: s.quantile(0.25),
        avg="mean", p75=lambda s: s.quantile(0.75), p90=lambda s: s.quantile(0.90), hi="max",
    ).sort_index()
    if band_smooth > 1:
        band = band.rolling(band_smooth, center=True, min_periods=1).mean()

    fig = go.Figure()
    # Colours match the Cotton On-Call seasonality chart.
    band_pairs = [("lo", "hi", "rgba(31,138,156,0.08)", "Min–Max"), ("p10", "p90", "rgba(31,138,156,0.16)", "10th–90th pct"), ("p25", "p75", "rgba(31,138,156,0.28)", "25th–75th pct")]
    for lo_col, hi_col, color, name in band_pairs:
        fig.add_trace(go.Scatter(x=band.index, y=band[hi_col], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=band.index, y=band[lo_col], fill="tonexty", fillcolor=color, line=dict(width=0), name=name))
    fig.add_trace(go.Scatter(x=band.index, y=band["avg"], mode="lines", name="Average", line=dict(color="#4a5578", width=1.5, dash="dot")))

    for yr, color, width in [(last_year, "#c94a4a", 2), (current_year, NAVY, 3)]:
        grp = long_df[long_df["yr"] == yr].sort_values(x_col)
        if grp.empty:
            continue
        fig.add_trace(go.Scatter(x=grp[x_col], y=grp[val_col], mode="lines", name=str(yr), line=dict(color=color, width=width)))

    xaxis = dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), title=dict(text=xaxis_title, font=dict(color=MUTED, size=11)), hoverformat=".1f")
    if reversed_x:
        xaxis["autorange"] = "reversed"
    base_layout(
        fig, title=title, xaxis=xaxis,
        yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), hoverformat=".1f",
                   title=dict(text=yaxis_title, font=dict(color=MUTED, size=11))),
    )
    return fig

# ── Header + tab selector ───────────────────────────────────────────────────────

st.markdown(
    "<style>"
    ".block-container{padding-top:3.5rem;padding-bottom:1rem}"
    # Tab bar + radios styled to match the Cotton On-Call dashboard: light-grey
    # rounded track, fully-rounded pills, navy fill + white text when selected.
    "div[data-testid='stButtonGroup'] div[data-baseweb='button-group']{"
    "background-color:#eef0f6;border-radius:999px;padding:4px;gap:4px;}"
    "button[data-testid^='stBaseButton-segmented_control']{"
    "border:none!important;border-radius:999px!important;"
    "padding:8px 20px;background-color:transparent!important;box-shadow:none!important;"
    "transition:background-color 0.15s ease;}"
    "button[data-testid^='stBaseButton-segmented_control'] p{font-size:14px;font-weight:400;}"
    "button[data-testid='stBaseButton-segmented_control'] p{color:#5a6688!important;}"
    "button[data-testid='stBaseButton-segmented_controlActive']{"
    f"background-color:{NAVY}!important;}}"
    "button[data-testid='stBaseButton-segmented_controlActive'] p{"
    "color:#ffffff!important;}"
    # radios -> same pill/segmented look as Cotton's date-range radio
    "div[role='radiogroup']{background:#eef0f6;padding:4px;border-radius:999px;gap:2px;"
    "display:inline-flex;flex-wrap:wrap;}"
    "div[role='radiogroup'] label{background:transparent!important;border-radius:999px!important;"
    "padding:4px 12px!important;margin:0!important;}"
    "div[role='radiogroup'] label[data-baseweb='radio']>div:first-child{display:none;}"
    "div[role='radiogroup'] label div[data-testid='stMarkdownContainer'] p{font-size:12px!important;color:#5a6688;}"
    "div[role='radiogroup'] label:has(input:checked){background:#0a2463!important;}"
    "div[role='radiogroup'] label:has(input:checked) div[data-testid='stMarkdownContainer'] p{"
    "color:#ffffff!important;font-weight:600;}"
    # sidebar title (replaces the old main-area header line)
    ".sb-title{font-family:'Fraunces',Georgia,serif;font-size:1.5rem;font-weight:600;color:#0a2463;margin-bottom:2px;}"
    ".sb-caption{font-size:11px;color:#7a86a8;margin-bottom:16px;line-height:1.4;}"
    "</style>",
    unsafe_allow_html=True,
)

page = st.segmented_control(
    "View", ["Spread Monitor", "Contract Explorer", "Term Structure"],
    default="Spread Monitor", key="page", label_visibility="collapsed",
)
page = page or "Spread Monitor"

# Full date extent across all price data (period presets count back from DATA_MAX).
_mins = [front[n].index.min() for n in front]
_maxs = [front[n].index.max() for n in front]
if contract_db_available:
    _mins += [contract_db[n]["Date"].min() for n in contract_db]
    _maxs += [contract_db[n]["Date"].max() for n in contract_db]
DATA_MIN, DATA_MAX = pd.Timestamp(min(_mins)), pd.Timestamp(max(_maxs))

period, d_start, d_end = "5Y", None, None

def render_period(box, n_all_yrs: int):
    """Global Period radio (5Y / 10Y / All (N yrs) / Custom). Drives both the date
    window of the time-series charts and which past years feed the seasonality
    bands/average. Returns (period, d_start, d_end)."""
    opts = ["5Y", "10Y", "All", "Custom"]
    with box:
        # choice remembered manually: the "All" label differs per tab, which would
        # otherwise look like a new widget to Streamlit and reset it on tab switch
        choice = st.radio(
            "Period", opts, index=opts.index(st.session_state.get("period_sel", "5Y")),
            horizontal=True, key=f"period_{page}", label_visibility="collapsed",
            format_func=lambda v: f"All ({n_all_yrs} yrs)" if v == "All" else v,
        )
        st.session_state["period_sel"] = choice
        if choice == "Custom":
            c1, c2 = st.columns(2)
            f = c1.date_input("From", value=max(DATA_MIN, DATA_MAX - pd.DateOffset(years=3)).date(),
                              min_value=DATA_MIN.date(), max_value=DATA_MAX.date(), key="period_from")
            t = c2.date_input("To", value=DATA_MAX.date(),
                              min_value=DATA_MIN.date(), max_value=DATA_MAX.date(), key="period_to")
            return choice, *sorted([f, t])
    if choice == "All":
        return choice, DATA_MIN.date(), DATA_MAX.date()
    return choice, (DATA_MAX - pd.DateOffset(years={"5Y": 5, "10Y": 10}[choice])).date(), DATA_MAX.date()

# ── Sidebar — shared controls ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div class='sb-title'>ARB Monitor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-caption'>KC/RC (Arabica vs Robusta) and CC/LCC (NY vs London Cocoa) spreads.</div>", unsafe_allow_html=True)
    st.markdown("### Configuration")
    pair = st.radio("Pair", ["KC / RC", "CC / LCC"],
                    index=0, horizontal=True, label_visibility="collapsed")
    pair_key = "KCRC" if pair.startswith("KC") else "CCLCC"

    if pair_key == "KCRC":
        st.divider()
        st.markdown("**Units**")
        unit_choice = st.radio("Units", ["$/MT", "¢/lb"], index=1, horizontal=True, label_visibility="collapsed")
    else:
        unit_choice = "$/MT"

    # Global period radio lives here but is filled in by render_period() once the
    # tab has built its series, so "All (N yrs)" counts that tab's real data.
    period_box = None
    if page != "Term Structure":
        st.divider()
        st.markdown("**Period**")
        period_box = st.container()

month_map = KCRC_MONTH_MAP if pair_key == "KCRC" else CCLCC_MONTH_MAP
pair_name_short = "KC / RC  —  Arabica vs Robusta" if pair_key == "KCRC" else "CC / LCC  —  NY vs London Cocoa"

def fx_note():
    """Cocoa only: say how LCC (quoted in GBP) is brought to USD for the spread."""
    if pair_key == "CCLCC":
        st.caption(r"Cocoa spread = CC (\$/MT) − LCC (£/MT) × GBP/USD, so both legs are in \$/MT. GBP/USD chart is at the bottom.")

def gbp_fx_chart():
    """Cocoa only: GBP/USD line over the chosen Period (last 5 years on Term Structure,
    which has no period)."""
    if pair_key != "CCLCC":
        return
    g = gbp_full.sort_index()
    if d_start is None:
        lo, hi = (g.index.max() - pd.DateOffset(years=5)).date(), g.index.max().date()
    else:
        lo, hi = d_start, d_end
    g = g.loc[str(lo):str(hi)]
    fig = go.Figure(go.Scatter(x=g.index, y=g, name="GBP/USD", line=dict(color=TEAL, width=1.5)))
    base_layout(fig, title=f"GBP/USD — used to convert LCC to USD (latest {g.iloc[-1]:.4f})" if len(g) else "GBP/USD",
                yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), hoverformat=".4f"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Spread Monitor  (rolled front-month, 1st/2nd nearby)
# ══════════════════════════════════════════════════════════════════════════════

if page == "Spread Monitor":

    with st.sidebar:
        st.divider()
        st.markdown("**Price source**")
        contract_choice = st.radio("Contract", ["1st month", "2nd month"],
                                   index=0, horizontal=True, label_visibility="collapsed")
        use_col = "px1" if "1st" in contract_choice else "px2"

        st.divider()
        st.markdown("**Windows**")
        zscore_win = st.slider("Z-score lookback (days)", 60, 504, 252, step=21)

    def _pick(name: str) -> pd.Series:
        return front[name][use_col].rename(name)

    src_tag = contract_choice.split("(")[0].strip()  # e.g. "1st month"

    if pair_key == "KCRC":
        kc_s    = _pick("KC")
        rc_s    = _pick("RC")
        kc_mt   = kc_s * KC_FACTOR
        spread  = (kc_mt - rc_s).dropna()
        if unit_choice == "¢/lb":
            spread = spread / KC_FACTOR
        leg1_label, leg2_label = f"KC ({unit_choice})", f"RC ({unit_choice})"
        spread_label = f"Arabica Premium over Robusta ({unit_choice})"
        pair_title   = f"{pair_name_short}  [{src_tag}]"
        has_fx       = False
    else:
        cc_s    = _pick("CC")
        lcc_s   = _pick("LCC")
        lcc_usd = (lcc_s * gbp_raw).dropna()
        spread  = (cc_s - lcc_usd).dropna()
        leg1_label, leg2_label = "CC ($/MT)", "LCC in USD ($/MT)"
        spread_label = "NY Premium over London Cocoa ($/MT)"
        pair_title   = f"{pair_name_short}  [{src_tag}]"
        has_fx       = True

    spread_full = spread.copy()  # full history, independent of the date-range slider below — seasonality needs every year

    period, d_start, d_end = render_period(period_box, spread.index.year.nunique() - 1)

    # ── Compute on the full history, then slice to the chosen period for display —
    #    a 252d rolling window computed over a 6M slice would be all NaN. ────────

    z_full   = zscore(spread, zscore_win)
    mu_full  = spread.rolling(zscore_win).mean()
    sig_full = spread.rolling(zscore_win).std()

    # l1 / l2 in the chosen units — used by all sections
    if pair_key == "KCRC":
        l1_full = _pick("KC") * KC_FACTOR
        l2_full = _pick("RC")
        if unit_choice == "¢/lb":
            l1_full = l1_full / KC_FACTOR
            l2_full = l2_full / KC_FACTOR
    else:
        l1_full = _pick("CC")
        l2_full = (_pick("LCC") * gbp_raw).dropna()

    def _view(x):
        return x.loc[str(d_start):str(d_end)]

    spread, z, mu, sig = _view(spread), _view(z_full), _view(mu_full), _view(sig_full)
    l1, l2 = _view(l1_full), _view(l2_full)

    fx_note()

    # ── SECTION 1 — Spread Monitor ───────────────────────────────────────────────

    # — Spread + bands —
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu + 2*sig, name="+2σ",
        line=dict(color=RED, width=1, dash="dot"), showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu + sig, name="+1σ",
        line=dict(color=AMBER, width=1, dash="dash"), showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu, name="Mean",
        line=dict(color=MUTED, width=1.5), showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu - sig, name="-1σ",
        line=dict(color=AMBER, width=1, dash="dash"), showlegend=False))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu - 2*sig, name="-2σ",
        line=dict(color=RED, width=1, dash="dot"), showlegend=False))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=spread, name="Spread",
        line=dict(color=TEAL, width=2), showlegend=True))
    base_layout(fig_sp, title=f"{spread_label} · {src_tag}")
    st.plotly_chart(fig_sp, use_container_width=True)

    # — Seasonality —
    season_df = spread_full.rename("val").to_frame()
    season_df["x"] = season_df.index.dayofyear
    season_df["yr"] = season_df.index.year
    cur_yr = season_df["yr"].max()
    hist_years = sorted(y for y in season_df["yr"].unique() if y < cur_yr)
    band_years = lookback_years(hist_years)
    fig_season = seasonality_bands_fig(
        season_df, "x", "val", f"Seasonality — {spread_label}",
        "Day of year", spread_label, cur_yr, cur_yr - 1, band_years=band_years,
    )
    st.plotly_chart(fig_season, use_container_width=True)

    # — Z-score —
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=z.index, y=z, name="Z-score",
        line=dict(color=TEAL, width=1.5)))
    fig_z.add_hline(y=0, line_color=MUTED, line_width=1)
    base_layout(fig_z, title=f"Z-Score  ({zscore_win}d rolling)",
                yaxis=dict(gridcolor=GRID, linecolor=GRID,
                           tickfont=dict(color=MUTED), range=[-4, 4]))
    st.plotly_chart(fig_z, use_container_width=True)

    # — Individual legs —
    fig_legs = go.Figure()
    fig_legs.add_trace(go.Scatter(x=l1.index, y=l1, name=leg1_label,
                                  line=dict(color=TEAL, width=1.5)))
    fig_legs.add_trace(go.Scatter(x=l2.index, y=l2, name=leg2_label,
                                  line=dict(color=AMBER, width=1.5)))
    base_layout(fig_legs, title=f"Individual Legs ({unit_choice})",
                yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED)))
    st.plotly_chart(fig_legs, use_container_width=True)

    st.divider()

    # ── SECTION 2 — Return Scatter ────────────────────────────────────────────────

    st.subheader("Return Scatter")
    dl1  = l1.diff().dropna()
    dl2  = l2.diff().dropna()
    scat = pd.concat([dl1.rename("leg1"), dl2.rename("leg2")], axis=1).dropna()

    if len(scat) < 10:
        st.info("Not enough data in the selected date range.")
    else:
        coeffs           = np.polyfit(scat["leg1"], scat["leg2"], 1)
        slope, intercept = coeffs
        x_line           = np.linspace(scat["leg1"].min(), scat["leg1"].max(), 200)
        y_line           = slope * x_line + intercept
        r2               = scat["leg1"].corr(scat["leg2"]) ** 2

        cutoff   = 60
        old_mask = scat.index < scat.index[-min(cutoff, len(scat))]
        recent   = scat[~old_mask]
        history  = scat[old_mask]

        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=history["leg1"], y=history["leg2"], mode="markers", name="History",
            marker=dict(color=MUTED, size=4, opacity=0.45),
            hovertemplate=f"Δ{leg1_label}: %{{x:.1f}}<br>Δ{leg2_label}: %{{y:.1f}}<extra></extra>",
        ))
        fig_scat.add_trace(go.Scatter(
            x=recent["leg1"], y=recent["leg2"], mode="markers",
            name=f"Last {min(cutoff, len(scat))}d",
            marker=dict(color=TEAL, size=6, opacity=0.85, line=dict(color="white", width=0.5)),
            hovertemplate=f"Δ{leg1_label}: %{{x:.1f}}<br>Δ{leg2_label}: %{{y:.1f}}<extra></extra>",
        ))
        fig_scat.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines", name="Regression",
            line=dict(color=RED, width=1.5, dash="dash"),
        ))

        # — Callouts for the most recent few sessions —
        callout_n = 5
        latest    = scat.iloc[-callout_n:]
        fig_scat.add_trace(go.Scatter(
            x=latest["leg1"], y=latest["leg2"], mode="markers",
            name=f"Last {callout_n} sessions",
            marker=dict(color=RED, size=10, symbol="circle-open", line=dict(color=RED, width=2)),
            hovertemplate=f"Δ{leg1_label}: %{{x:.1f}}<br>Δ{leg2_label}: %{{y:.1f}}<extra></extra>",
        ))
        for i, (dt, row) in enumerate(latest.iterrows()):
            fig_scat.add_annotation(
                x=row["leg1"], y=row["leg2"], text=dt.strftime("%d %b"),
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=RED,
                ax=0, ay=-28 - (i % 2) * 16,
                font=dict(size=10, color=FONT),
                bgcolor="rgba(255,255,255,0.85)", bordercolor=RED, borderwidth=1, borderpad=2,
            )

        fig_scat.add_hline(y=0, line_color=GRID, line_width=1)
        fig_scat.add_vline(x=0, line_color=GRID, line_width=1)
        base_layout(
            fig_scat,
            title=f"Daily Return Scatter  —  R²={r2:.2f}",
            xaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                       title=dict(text=f"Δ {leg1_label}", font=dict(color=MUTED, size=11))),
            yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                       title=dict(text=f"Δ {leg2_label}", font=dict(color=MUTED, size=11))),
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    st.divider()

    # ── SECTION 3 — Ratio (KC/RC only) ───────────────────────────────────────────

    if not has_fx:
        st.subheader("Ratio")
        st.caption("Ratio of Arabica to Robusta price. "
                   "Roasters blend the two; extreme ratios historically mean-revert "
                   "as substitution economics kick in.")

        ratio_full = l1_full / l2_full
        ratio = _view(ratio_full)
        mu_r  = _view(ratio_full.rolling(zscore_win).mean())
        sig_r = _view(ratio_full.rolling(zscore_win).std())

        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r + sig_r,
                                       line=dict(color=AMBER, width=1, dash="dash"), name="+1σ"))
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r - sig_r,
                                       line=dict(color=AMBER, width=1, dash="dash"),
                                       name="-1σ", showlegend=False))
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r,
                                       line=dict(color=MUTED, width=1), name="Mean"))
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=ratio,
                                       line=dict(color=TEAL, width=2), name="KC/RC Ratio"))
        base_layout(fig_ratio, title="KC/RC Price Ratio (Arabica/Robusta)")
        st.plotly_chart(fig_ratio, use_container_width=True)

    gbp_fx_chart()
    st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) front-month (1st/2nd) + GBP/USD")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Contract Explorer  (specific-vintage time series)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Contract Explorer":

    fx_note()

    anchor_month = None
    with st.sidebar:
        if contract_db_available:
            st.divider()
            st.markdown("**Contract Explorer**")
            anchor_month = st.radio("Anchor month", list(month_map.keys()),
                                    horizontal=True, key=f"anchor_month_{pair_key}")
            st.divider()
            st.markdown("**Windows**")
            zscore_win2 = st.slider("Z-score lookback (days)", 60, 504, 252, step=21, key="zscore_win2")

    if not contract_db_available:
        st.info("Per-contract data not yet synced — run ingest_contracts.py first.")
    else:
        m1, m2, yoff2 = month_map[anchor_month]
        leg1_name, leg2_name = ("KC", "RC") if pair_key == "KCRC" else ("CC", "LCC")
        db1 = contract_db["KC"] if pair_key == "KCRC" else contract_db["CC"]
        db2 = contract_db["RC"] if pair_key == "KCRC" else contract_db["LCC"]

        merged = continuous_pair(db1, m1, db2, m2, yoff2)

        if merged.empty:
            st.info("No overlapping data for this anchor month.")
        else:
            st.caption(
                f"Anchor month **{anchor_month}** → {leg1_name} {m1} vs {leg2_name} {m2}"
                + (f" (+{yoff2}y)" if yoff2 else "")
                + " — one continuous line. It moves to the next contract right after each one expires."
            )

            leg1 = merged["leg1"]
            if pair_key == "KCRC":
                leg1c = leg1 * KC_FACTOR
                leg2c = merged["leg2"].copy()
                if unit_choice == "¢/lb":
                    leg1c, leg2c = leg1c / KC_FACTOR, leg2c / KC_FACTOR
            else:
                leg2c = (merged["leg2"] * gbp_full.reindex(merged.index).ffill()).dropna()
                leg1c = leg1.reindex(leg2c.index)

            spr = (leg1c - leg2c).dropna()
            unit_lbl = unit_choice if pair_key == "KCRC" else "$/MT"

            period, d_start, d_end = render_period(period_box, merged["year1"].nunique() - 1)

            tag1 = leg1_name + m1 + merged["year1"].astype(str).str[-2:]
            tag2 = leg2_name + m2 + merged["year2"].astype(str).str[-2:]

            mu2  = spr.rolling(zscore_win2).mean()
            sig2 = spr.rolling(zscore_win2).std()

            def _v(x):
                return x.loc[str(d_start):str(d_end)]
            spr_v, mu2_v, sig2_v = _v(spr), _v(mu2), _v(sig2)

            # — Spread + bands —
            fig_sp2 = go.Figure()
            fig_sp2.add_trace(go.Scatter(x=spr_v.index, y=mu2_v + 2*sig2_v, name="+2σ",
                                         line=dict(color=RED, width=1, dash="dot")))
            fig_sp2.add_trace(go.Scatter(x=spr_v.index, y=mu2_v + sig2_v, name="+1σ",
                                         line=dict(color=AMBER, width=1, dash="dash")))
            fig_sp2.add_trace(go.Scatter(x=spr_v.index, y=mu2_v, name="Mean",
                                         line=dict(color=MUTED, width=1.5)))
            fig_sp2.add_trace(go.Scatter(x=spr_v.index, y=mu2_v - sig2_v, name="-1σ",
                                         line=dict(color=AMBER, width=1, dash="dash"), showlegend=False))
            fig_sp2.add_trace(go.Scatter(x=spr_v.index, y=mu2_v - 2*sig2_v, name="-2σ",
                                         line=dict(color=RED, width=1, dash="dot"), showlegend=False))
            fig_sp2.add_trace(go.Scatter(x=spr_v.index, y=spr_v, name="Spread",
                                         line=dict(color=TEAL, width=2)))
            base_layout(fig_sp2, title=f"Spread — anchor {anchor_month} ({unit_lbl})")
            st.plotly_chart(fig_sp2, use_container_width=True)

            # — Seasonality (by days to expiry, not calendar day - a vintage's own
            # lifecycle is what's comparable year to year here, not the date it fell on) —
            ltd_map = {yr: get_ltd(db1, m1, yr) for yr in merged["year1"].unique()}
            season_df2 = spr.rename("val").to_frame()
            season_df2["year1"] = merged["year1"].reindex(season_df2.index)
            season_df2["ltd"] = season_df2["year1"].map(ltd_map)
            season_df2 = season_df2.dropna(subset=["ltd"])
            season_df2["x"] = (season_df2["ltd"] - season_df2.index).dt.days
            season_df2["yr"] = season_df2["year1"].astype(int)
            cur_vintage = int(merged["year1"].iloc[-1])
            hist_vintages = sorted(y for y in season_df2["yr"].unique() if y < cur_vintage)

            lb_col, dte_col = st.columns([3, 2])
            with lb_col:
                band_years2 = lookback_years(hist_vintages)
            with dte_col:
                # Nearly every vintage only trades in this leg for its final ~364 days
                # (it starts once the previous vintage expires) - only the very first
                # vintage runs to 600-700 days, which is what left the chart's left
                # side blank. 365 shows one full lifecycle.
                dte_max = st.number_input("Days to expiry (max)", min_value=30, max_value=730,
                                          value=365, step=30, key="ce_dte_max")
            season_df2 = season_df2[season_df2["x"] <= dte_max]

            fig_season2 = seasonality_bands_fig(
                season_df2, "x", "val", f"Seasonality — anchor {anchor_month} ({unit_lbl})",
                "Days to expiry", f"Spread ({unit_lbl})", cur_vintage, cur_vintage - 1,
                reversed_x=True, band_years=band_years2,
            )
            st.plotly_chart(fig_season2, use_container_width=True)

            # — Z-score —
            z2 = zscore(spr, zscore_win2)
            fig_z2 = go.Figure()
            z2_v = _v(z2)
            fig_z2.add_trace(go.Scatter(x=z2_v.index, y=z2_v, name="Z-score", line=dict(color=TEAL, width=1.5)))
            fig_z2.add_hline(y=0, line_color=MUTED, line_width=1)
            base_layout(fig_z2, title=f"Z-Score  ({zscore_win2}d rolling)",
                        yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), range=[-4, 4]))
            st.plotly_chart(fig_z2, use_container_width=True)

            # — Individual legs —
            fig_legs2 = go.Figure()
            leg1_v, leg2_v = _v(leg1c), _v(leg2c)
            fig_legs2.add_trace(go.Scatter(
                x=leg1_v.index, y=leg1_v, name=f"{leg1_name} {m1} (continuous)",
                line=dict(color=TEAL, width=1.5),
                customdata=tag1.reindex(leg1_v.index), hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>"))
            fig_legs2.add_trace(go.Scatter(
                x=leg2_v.index, y=leg2_v, name=f"{leg2_name} {m2} (continuous)",
                line=dict(color=AMBER, width=1.5),
                customdata=tag2.reindex(leg2_v.index), hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>"))
            base_layout(fig_legs2, title=f"Individual Legs ({unit_lbl})",
                        yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED)))
            st.plotly_chart(fig_legs2, use_container_width=True)

            st.divider()
            st.caption("Roll schedule — which vintage was active over which real dates")
            roll_rows = []
            for yr, grp in merged.groupby("year1"):
                y2 = yr + yoff2
                roll_rows.append({
                    "Vintage":          yr,
                    f"{leg1_name} leg": f"{leg1_name}{m1}{str(yr)[-2:]}",
                    f"{leg2_name} leg": f"{leg2_name}{m2}{str(y2)[-2:]}",
                    "From":             grp.index.min().date(),
                    "To":               grp.index.max().date(),
                })
            st.dataframe(pd.DataFrame(roll_rows), hide_index=True, use_container_width=True)

    gbp_fx_chart()
    st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) per-contract (KC/RC/CC/LCC)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Term Structure  (all active/upcoming spreads, one date)
# ══════════════════════════════════════════════════════════════════════════════

else:

    fx_note()

    z_choice = None
    with st.sidebar:
        if contract_db_available:
            st.divider()
            st.markdown("**Term Structure**")

            _db1 = contract_db["KC"] if pair_key == "KCRC" else contract_db["CC"]
            _db2 = contract_db["RC"] if pair_key == "KCRC" else contract_db["LCC"]
            _date_min = min(_db1["Date"].min(), _db2["Date"].min())
            _date_max = max(_db1["Date"].max(), _db2["Date"].max())

            asof_date = st.date_input(
                "As of date", value=_date_max.date(),
                min_value=_date_min.date(), max_value=_date_max.date(),
                key="ts_asof",
            )

            if pair_key == "KCRC":
                z_choice = st.radio(
                    "Z-month pairing", ["ZX", "ZF"], horizontal=True, key="ts_z_choice",
                    format_func=lambda v: "ZX (same yr)" if v == "ZX" else "ZF (next yr)",
                    help="ZX = KC Z vs RC X, same year. ZF = KC Z vs RC F, next year.",
                )

            n_maturities = st.slider("Number of maturities", 4, 16, 8, step=1, key="ts_n")

    if not contract_db_available:
        st.info("Per-contract data not yet synced — run ingest_contracts.py first.")
    else:
        leg1_name, leg2_name = ("KC", "RC") if pair_key == "KCRC" else ("CC", "LCC")
        db1 = contract_db["KC"] if pair_key == "KCRC" else contract_db["CC"]
        db2 = contract_db["RC"] if pair_key == "KCRC" else contract_db["LCC"]

        if pair_key == "KCRC":
            anchor_seq = ["H", "K", "N", "U", z_choice]
        else:
            anchor_seq = list(CCLCC_MONTH_MAP.keys())

        asof_ts = pd.Timestamp(asof_date)
        term_rows = build_term_structure(db1, month_map, anchor_seq, db2, asof_ts, n_maturities)

        if not term_rows:
            st.info("No active or upcoming contracts found as of this date.")
        else:
            st.caption(
                f"All listed {leg1_name}/{leg2_name} maturities not yet expired as of "
                f"**{asof_ts.date()}**, in order — this is a snapshot on one date, not a "
                "time series like the other two tabs."
            )

            tags, spreads, customdata = [], [], []
            for r in term_rows:
                if pair_key == "KCRC":
                    p1c = r["price1"] * KC_FACTOR
                    p2c = r["price2"]
                    if unit_choice == "¢/lb":
                        p1c, p2c = p1c / KC_FACTOR, p2c / KC_FACTOR
                else:
                    p2c = r["price2"] * gbp_full.sort_index().asof(r["date2"])
                    p1c = r["price1"]

                tag1 = f"{leg1_name}{r['m1']}{str(r['year'])[-2:]}"
                tag2 = f"{leg2_name}{r['m2']}{str(r['y2'])[-2:]}"
                tags.append(f"{r['anchor']}{str(r['year'])[-2:]}")
                spreads.append(p1c - p2c)
                stale1 = "" if r["date1"] == asof_ts.normalize() else f"  (last quote {r['date1'].date()})"
                stale2 = "" if r["date2"] == asof_ts.normalize() else f"  (last quote {r['date2'].date()})"
                customdata.append(f"{tag1}: {p1c:,.2f}{stale1}<br>{tag2}: {p2c:,.2f}{stale2}")

            unit_lbl = unit_choice if pair_key == "KCRC" else "$/MT"
            bar_colors = [GREEN if v >= 0 else RED for v in spreads]

            # Bars default to a 0-based axis, which flattens exactly the kind of
            # small maturity-to-maturity differences a term structure is meant to
            # show. Auto-scale to the data instead, with some headroom — bars can
            # end up not touching the bottom of the plot, but the shape of the
            # curve (what actually matters here) becomes visible.
            y_min, y_max = min(spreads), max(spreads)
            pad = max((y_max - y_min) * 0.15, abs(y_max) * 0.02, 1.0)
            y_range = [y_min - pad, y_max + pad]

            fig_term = go.Figure()
            fig_term.add_trace(go.Bar(
                x=tags, y=spreads, marker_color=bar_colors, opacity=0.85,
                customdata=customdata,
                hovertemplate="<b>%{x}</b><br>Spread: %{y:.2f}<br>%{customdata}<extra></extra>",
            ))
            fig_term.add_trace(go.Scatter(
                x=tags, y=spreads, mode="lines+markers", name="Curve",
                line=dict(color=FONT, width=1.5), marker=dict(size=6, color=FONT),
                hoverinfo="skip", showlegend=False,
            ))
            if y_range[0] <= 0 <= y_range[1]:
                fig_term.add_hline(y=0, line_color=MUTED, line_width=1)
            base_layout(
                fig_term,
                title=f"{leg1_name}/{leg2_name} Term Structure — as of {asof_ts.date()} ({unit_lbl})",
                xaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                           type="category", categoryorder="array", categoryarray=tags),
                yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), range=y_range),
                showlegend=False,
            )
            st.plotly_chart(fig_term, use_container_width=True)

            table_rows = []
            for r, tag, spr_v in zip(term_rows, tags, spreads):
                table_rows.append({
                    "Maturity":         tag,
                    f"{leg1_name} leg": f"{leg1_name}{r['m1']}{str(r['year'])[-2:]}",
                    f"{leg2_name} leg": f"{leg2_name}{r['m2']}{str(r['y2'])[-2:]}",
                    "Spread":           round(spr_v, 2),
                    "Expiry":           r["ltd1"].date(),
                })
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    gbp_fx_chart()
    st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) per-contract (KC/RC/CC/LCC)")
