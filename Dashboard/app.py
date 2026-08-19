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
def load_all():
    gbp = pd.read_parquet(DB / "fx_gbp.parquet")["GBP_USD"]

    # Actual front-month prices (1st/2nd month, no roll adjustment)
    front = {}
    for name in ["KC", "RC", "CC", "LCC"]:
        path = DB / f"front_{name}.parquet"
        front[name] = pd.read_parquet(path) if path.exists() else None

    return gbp, front

gbp_raw, front = load_all()

front_available = all(front[n] is not None for n in ["KC", "RC", "CC", "LCC"])

if not front_available:
    st.error("Front-month data not yet ingested — run ingest_front.py first.")
    st.stop()

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

    st.divider()
    st.markdown("**Price source**")
    contract_choice = st.radio("Contract", ["1st month (actual)", "2nd month (actual)"],
                               index=0, label_visibility="collapsed")
    use_col = "px1" if "1st" in contract_choice else "px2"

    st.divider()
    st.markdown("**Windows**")
    zscore_win = st.slider("Z-score lookback (days)", 60, 504, 252, step=21)

    st.divider()
    st.markdown("**Signal thresholds**")
    z_entry = st.number_input("Entry z-score", value=1.5, step=0.1, format="%.1f")
    z_exit  = st.number_input("Exit z-score",  value=0.5, step=0.1, format="%.1f")

# ── Build spread ──────────────────────────────────────────────────────────────

def _pick(name: str) -> pd.Series:
    return front[name][use_col].rename(name)

src_tag = contract_choice.split("(")[0].strip()  # e.g. "1st month"

if pair_key == "KCRC":
    kc_s    = _pick("KC")
    rc_s    = _pick("RC")
    kc_mt   = kc_s * KC_FACTOR
    spread  = (kc_mt - rc_s).dropna()
    leg1_label, leg2_label = "KC ($/MT)", "RC ($/MT)"
    spread_label = "Arabica Premium over Robusta ($/MT)"
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

st.title("ARB Monitor")
st.caption(pair_title)
st.markdown("**Date range**")

d_start, d_end = st.slider(
    "range", min_value=date_min, max_value=date_max,
    value=(date_min, date_max), format="DD MMM YYYY",
    label_visibility="collapsed",
)
cal_l, cal_r, _ = st.columns([1, 1, 4])
with cal_l:
    d_start = st.date_input("Start", value=d_start, min_value=date_min,
                             max_value=date_max, key="ds")
with cal_r:
    d_end   = st.date_input("End",   value=d_end,   min_value=date_min,
                             max_value=date_max, key="de")

spread  = spread.loc[str(d_start): str(d_end)]
gbp_raw = gbp_raw.loc[str(d_start): str(d_end)]

st.divider()

# ── Compute ───────────────────────────────────────────────────────────────────

z   = zscore(spread, zscore_win)
mu  = spread.rolling(zscore_win).mean()
sig = spread.rolling(zscore_win).std()

# l1 / l2 in $/MT — used by all sections
if pair_key == "KCRC":
    l1 = (_pick("KC") * KC_FACTOR).loc[str(d_start):str(d_end)]
    l2 = _pick("RC").loc[str(d_start):str(d_end)]
else:
    l1 = _pick("CC").loc[str(d_start):str(d_end)]
    l2 = (_pick("LCC") * gbp_raw).dropna().loc[str(d_start):str(d_end)]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Return Scatter
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("Section 1 — Return Scatter", expanded=True):
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Spread Monitor
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("Section 2 — Spread Monitor", expanded=True):
    st.caption("Spread level with rolling mean and 1/2 standard deviation bands. "
               "The z-score panel shows where the spread sits relative to its own history.")

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
    fig_z.add_hrect(y0=z_entry, y1=4,   fillcolor=RED,   opacity=0.06, line_width=0)
    fig_z.add_hrect(y0=-4, y1=-z_entry, fillcolor=GREEN, opacity=0.06, line_width=0)
    fig_z.add_trace(go.Scatter(
        x=z.index, y=z, name="Z-score",
        line=dict(color=TEAL, width=1.5)))
    for level, color in [(z_entry, RED), (-z_entry, GREEN), (z_exit, AMBER), (-z_exit, AMBER)]:
        fig_z.add_hline(y=level, line_dash="dot", line_color=color, line_width=1)
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
    base_layout(fig_legs, title="Individual Legs ($/MT)",
                yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED)))
    st.plotly_chart(fig_legs, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Ratio (KC/RC only)
# ══════════════════════════════════════════════════════════════════════════════

if not has_fx:
    with st.expander("Section 3 — Ratio", expanded=True):
        st.caption("Ratio of Arabica to Robusta price (both in $/MT). "
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
        base_layout(fig_ratio, title="KC/RC Price Ratio (Arabica/Robusta, $/MT)")
        st.plotly_chart(fig_ratio, use_container_width=True)

st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) front-month (1st/2nd) + GBP/USD")
