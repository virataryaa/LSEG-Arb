"""
ICEBREAKER — ARB Dashboard
Spread Monitor for KC/RC and CC/LCC pairs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ARB Monitor", layout="wide")

DB = Path(__file__).parent.parent / "Database"

KC_FACTOR = 22.0462   # ¢/lbs  →  $/MT

# Contract-pairing convention for the Contract Explorer section, mirroring
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
TEAL   = "#2563eb"
GREEN  = "#16a34a"
RED    = "#dc2626"
AMBER  = "#d97706"

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

def contract_series(df: pd.DataFrame, month: str, year: int) -> pd.Series:
    sub = df[(df["month"] == month) & (df["year"] == year)].sort_values("Date")
    return sub.set_index("Date")["settlement"].dropna()

def available_vintages(df: pd.DataFrame, month: str, min_days: int = 20) -> list[int]:
    counts = df[df["month"] == month].groupby("year").size()
    return sorted(int(y) for y, n in counts.items() if n >= min_days)

# ── Analytics helpers ─────────────────────────────────────────────────────────

def zscore(spread: pd.Series, window: int) -> pd.Series:
    mu  = spread.rolling(window).mean()
    sig = spread.rolling(window).std()
    return (spread - mu) / sig

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Configuration")
    pair = st.radio("Pair", ["KC / RC  (Arabica vs Robusta)", "CC / LCC  (NY vs London Cocoa)"],
                    index=0, label_visibility="collapsed")
    pair_key = "KCRC" if pair.startswith("KC") else "CCLCC"

    if pair_key == "KCRC":
        st.divider()
        st.markdown("**Units**")
        unit_choice = st.radio("Units", ["$/MT", "¢/lb"], index=1, label_visibility="collapsed")
    else:
        unit_choice = "$/MT"

    st.divider()
    st.markdown("**Price source**")
    contract_choice = st.radio("Contract", ["1st month (actual)", "2nd month (actual)"],
                               index=0, label_visibility="collapsed")
    use_col = "px1" if "1st" in contract_choice else "px2"

    st.divider()
    st.markdown("**Windows**")
    zscore_win = st.slider("Z-score lookback (days)", 60, 504, 252, step=21)

    month_map = KCRC_MONTH_MAP if pair_key == "KCRC" else CCLCC_MONTH_MAP
    anchor_month, vintage_years = None, []
    if contract_db_available:
        st.divider()
        st.markdown("**Contract Explorer**")
        anchor_month = st.radio("Anchor month", list(month_map.keys()),
                                horizontal=True, key=f"anchor_month_{pair_key}")
        _m1, _m2, _yoff2 = month_map[anchor_month]
        _db1 = contract_db["KC"] if pair_key == "KCRC" else contract_db["CC"]
        _vintages = available_vintages(_db1, _m1)
        _default_vint = _vintages[-3:] if len(_vintages) >= 3 else _vintages
        vintage_years = st.multiselect("Vintage year(s)", options=_vintages,
                                       default=_default_vint, key=f"vintage_years_{pair_key}_{anchor_month}")

# ── Build spread ──────────────────────────────────────────────────────────────

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
    pair_title   = f"KC / RC  —  Arabica vs Robusta  [{src_tag}]"
    has_fx       = False
else:
    cc_s    = _pick("CC")
    lcc_s   = _pick("LCC")
    lcc_usd = (lcc_s * gbp_raw).dropna()
    spread  = (cc_s - lcc_usd).dropna()
    leg1_label, leg2_label = "CC ($/MT)", "LCC in USD ($/MT)"
    spread_label = "NY Premium over London Cocoa ($/MT)"
    pair_title   = f"CC / LCC  —  NY vs London Cocoa  [{src_tag}]"
    has_fx       = True

# ── Date range ────────────────────────────────────────────────────────────────

date_min = spread.index.min().date()
date_max = spread.index.max().date()

st.markdown(
    "<style>"
    ".block-container{padding-top:1.2rem;padding-bottom:1rem}"
    "header[data-testid='stHeader']{height:2rem}"
    "</style>"
    f"<div style='font-size:0.8rem;color:{MUTED};letter-spacing:0.04em;"
    f"text-transform:uppercase'>ARB Monitor &nbsp;·&nbsp; {pair_title}</div>",
    unsafe_allow_html=True,
)

def _dates_to_slider() -> None:
    """Push a manual Start/End edit back into the slider."""
    s, e = st.session_state.ds, st.session_state.de
    if s > e:
        s, e = e, s
    st.session_state.rng = (s, e)

# Single source of truth for the range; the slider owns "rng", the two date
# pickers mirror it. Keyed widgets ignore `value=` after first render, so the
# mirroring has to go through session state or the slider gets overruled.
if "rng" not in st.session_state:
    st.session_state.rng = (date_min, date_max)

s0, e0 = st.session_state.rng
s0 = min(max(s0, date_min), date_max)
e0 = min(max(e0, date_min), date_max)
if s0 > e0:
    s0, e0 = e0, s0
st.session_state.rng = (s0, e0)
st.session_state.ds  = s0
st.session_state.de  = e0

with st.sidebar:
    st.divider()
    st.markdown("**Date range**")
    st.slider(
        "range", min_value=date_min, max_value=date_max,
        format="DD MMM YYYY", key="rng",
        label_visibility="collapsed",
    )
    cal_l, cal_r = st.columns(2)
    with cal_l:
        st.date_input("Start", min_value=date_min, max_value=date_max,
                      key="ds", on_change=_dates_to_slider)
    with cal_r:
        st.date_input("End",   min_value=date_min, max_value=date_max,
                      key="de", on_change=_dates_to_slider)

d_start, d_end = st.session_state.rng

spread  = spread.loc[str(d_start): str(d_end)]
gbp_raw = gbp_raw.loc[str(d_start): str(d_end)]

# ── Compute ───────────────────────────────────────────────────────────────────

z   = zscore(spread, zscore_win)
mu  = spread.rolling(zscore_win).mean()
sig = spread.rolling(zscore_win).std()

# l1 / l2 in $/MT — used by all sections
if pair_key == "KCRC":
    l1 = (_pick("KC") * KC_FACTOR).loc[str(d_start):str(d_end)]
    l2 = _pick("RC").loc[str(d_start):str(d_end)]
    if unit_choice == "¢/lb":
        l1 = l1 / KC_FACTOR
        l2 = l2 / KC_FACTOR
else:
    l1 = _pick("CC").loc[str(d_start):str(d_end)]
    l2 = (_pick("LCC") * gbp_raw).dropna().loc[str(d_start):str(d_end)]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Spread Monitor
# ══════════════════════════════════════════════════════════════════════════════

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
base_layout(fig_sp, title=spread_label)
st.plotly_chart(fig_sp, use_container_width=True)

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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Return Scatter
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Ratio (KC/RC only)
# ══════════════════════════════════════════════════════════════════════════════

if not has_fx:
    st.subheader("Ratio")
    st.caption("Ratio of Arabica to Robusta price. "
               "Roasters blend the two; extreme ratios historically mean-revert "
               "as substitution economics kick in.")

    ratio = l1 / l2
    mu_r  = ratio.rolling(zscore_win).mean()
    sig_r = ratio.rolling(zscore_win).std()

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

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Contract Explorer (specific-vintage time series)
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Contract Explorer")

if not contract_db_available:
    st.info("Per-contract data not yet synced — run ingest_contracts.py first.")
elif not vintage_years:
    st.info("Select at least one vintage year in the sidebar.")
else:
    m1, m2, yoff2 = month_map[anchor_month]
    leg1_name, leg2_name = ("KC", "RC") if pair_key == "KCRC" else ("CC", "LCC")
    db1 = contract_db["KC"] if pair_key == "KCRC" else contract_db["CC"]
    db2 = contract_db["RC"] if pair_key == "KCRC" else contract_db["LCC"]

    st.caption(
        f"Anchor month **{anchor_month}** → {leg1_name} {m1} vs {leg2_name} {m2}"
        + (f" (+{yoff2}y)" if yoff2 else "")
        + " — real calendar dates for the specific contract vintage(s) selected, "
          "not the rolled front-month series used above."
    )

    YEAR_COLORS = px.colors.qualitative.Dark24

    fig_legs2   = go.Figure()
    fig_spread2 = go.Figure()
    rows        = []

    for i, yr in enumerate(sorted(vintage_years)):
        y2 = yr + yoff2
        s1 = contract_series(db1, m1, yr)
        s2 = contract_series(db2, m2, y2)
        if s1.empty or s2.empty:
            continue

        if pair_key == "KCRC":
            s1c = s1 * KC_FACTOR
            s2c = s2.copy()
            if unit_choice == "¢/lb":
                s1c, s2c = s1c / KC_FACTOR, s2c / KC_FACTOR
        else:
            s1c = s1.copy()
            s2c = (s2 * gbp_full.reindex(s2.index).ffill()).dropna()

        spr   = (s1c - s2c).dropna()
        color = YEAR_COLORS[i % len(YEAR_COLORS)]
        tag1  = f"{leg1_name}{m1}{str(yr)[-2:]}"
        tag2  = f"{leg2_name}{m2}{str(y2)[-2:]}"

        fig_legs2.add_trace(go.Scatter(x=s1c.index, y=s1c, name=tag1,
                                       line=dict(color=color, width=2)))
        fig_legs2.add_trace(go.Scatter(x=s2c.index, y=s2c, name=tag2,
                                       line=dict(color=color, width=1.5, dash="dot")))
        fig_spread2.add_trace(go.Scatter(x=spr.index, y=spr, name=str(yr),
                                         line=dict(color=color, width=2)))

        rows.append({
            "Vintage":     yr,
            f"{leg1_name} leg": tag1,
            f"{leg2_name} leg": tag2,
            "First trade": s1c.index.min().date(),
            "Last trade":  min(s1c.index.max(), s2c.index.max()).date(),
        })

    if not rows:
        st.info("No overlapping data for the selected vintage(s).")
    else:
        unit_lbl = unit_choice if pair_key == "KCRC" else "$/MT"
        base_layout(fig_legs2, title=f"Individual Legs by Vintage ({unit_lbl})")
        st.plotly_chart(fig_legs2, use_container_width=True)

        base_layout(fig_spread2, title=f"Spread by Vintage — anchor {anchor_month} ({unit_lbl})")
        st.plotly_chart(fig_spread2, use_container_width=True)

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) front-month (1st/2nd) + GBP/USD")
