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

# ── Header + tab selector ───────────────────────────────────────────────────────

st.markdown(
    "<style>"
    ".block-container{padding-top:3.5rem;padding-bottom:1rem}"
    # Pill-shaped tab bar: light-grey track, fully-rounded buttons, solid
    # blue fill + white bold text on the active one, muted grey on the rest.
    "div[data-testid='stButtonGroup'] div[data-baseweb='button-group']{"
    "background-color:#eef1f6;border-radius:999px;padding:4px;gap:2px;}"
    "button[data-testid^='stBaseButton-segmented_control']{"
    "border:none!important;border-radius:999px!important;"
    "font-size:1rem;padding:0.5rem 1.6rem;font-weight:500;"
    "background-color:transparent!important;box-shadow:none!important;"
    "transition:background-color 0.15s ease;}"
    "button[data-testid='stBaseButton-segmented_control'] p{color:#6b7280!important;}"
    "button[data-testid='stBaseButton-segmented_controlActive']{"
    f"background-color:{TEAL}!important;}}"
    "button[data-testid='stBaseButton-segmented_controlActive'] p{"
    "color:#ffffff!important;font-weight:600;}"
    "</style>",
    unsafe_allow_html=True,
)

page = st.segmented_control(
    "View", ["Spread Monitor", "Contract Explorer", "Term Structure"],
    default="Spread Monitor", key="page",
)
page = page or "Spread Monitor"

# ── Sidebar — shared controls ────────────────────────────────────────────────────

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

month_map = KCRC_MONTH_MAP if pair_key == "KCRC" else CCLCC_MONTH_MAP
pair_name_short = "KC / RC  —  Arabica vs Robusta" if pair_key == "KCRC" else "CC / LCC  —  NY vs London Cocoa"

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Spread Monitor  (rolled front-month, 1st/2nd nearby)
# ══════════════════════════════════════════════════════════════════════════════

if page == "Spread Monitor":

    with st.sidebar:
        st.divider()
        st.markdown("**Price source**")
        contract_choice = st.radio("Contract", ["1st month (actual)", "2nd month (actual)"],
                                   index=0, label_visibility="collapsed")
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

    # ── Date range ────────────────────────────────────────────────────────────

    date_min = spread.index.min().date()
    date_max = spread.index.max().date()

    st.markdown(
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

    # ── Compute ───────────────────────────────────────────────────────────────

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

    st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) front-month (1st/2nd) + GBP/USD")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Contract Explorer  (specific-vintage time series)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Contract Explorer":

    st.markdown(
        f"<div style='font-size:0.8rem;color:{MUTED};letter-spacing:0.04em;"
        f"text-transform:uppercase'>ARB Monitor &nbsp;·&nbsp; {pair_name_short}  [Contract Explorer]</div>",
        unsafe_allow_html=True,
    )

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

    st.subheader("Contract Explorer")

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

            tag1 = leg1_name + m1 + merged["year1"].astype(str).str[-2:]
            tag2 = leg2_name + m2 + merged["year2"].astype(str).str[-2:]

            mu2  = spr.rolling(zscore_win2).mean()
            sig2 = spr.rolling(zscore_win2).std()

            # — Spread + bands —
            fig_sp2 = go.Figure()
            fig_sp2.add_trace(go.Scatter(x=spr.index, y=mu2 + 2*sig2, name="+2σ",
                                         line=dict(color=RED, width=1, dash="dot")))
            fig_sp2.add_trace(go.Scatter(x=spr.index, y=mu2 + sig2, name="+1σ",
                                         line=dict(color=AMBER, width=1, dash="dash")))
            fig_sp2.add_trace(go.Scatter(x=spr.index, y=mu2, name="Mean",
                                         line=dict(color=MUTED, width=1.5)))
            fig_sp2.add_trace(go.Scatter(x=spr.index, y=mu2 - sig2, name="-1σ",
                                         line=dict(color=AMBER, width=1, dash="dash"), showlegend=False))
            fig_sp2.add_trace(go.Scatter(x=spr.index, y=mu2 - 2*sig2, name="-2σ",
                                         line=dict(color=RED, width=1, dash="dot"), showlegend=False))
            fig_sp2.add_trace(go.Scatter(x=spr.index, y=spr, name="Spread",
                                         line=dict(color=TEAL, width=2)))
            base_layout(fig_sp2, title=f"Spread — anchor {anchor_month} ({unit_lbl})")
            st.plotly_chart(fig_sp2, use_container_width=True)

            # — Z-score —
            z2 = zscore(spr, zscore_win2)
            fig_z2 = go.Figure()
            fig_z2.add_trace(go.Scatter(x=z2.index, y=z2, name="Z-score", line=dict(color=TEAL, width=1.5)))
            fig_z2.add_hline(y=0, line_color=MUTED, line_width=1)
            base_layout(fig_z2, title=f"Z-Score  ({zscore_win2}d rolling)",
                        yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), range=[-4, 4]))
            st.plotly_chart(fig_z2, use_container_width=True)

            # — Individual legs —
            fig_legs2 = go.Figure()
            fig_legs2.add_trace(go.Scatter(
                x=leg1c.index, y=leg1c, name=f"{leg1_name} {m1} (continuous)",
                line=dict(color=TEAL, width=1.5),
                customdata=tag1, hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>"))
            fig_legs2.add_trace(go.Scatter(
                x=leg2c.index, y=leg2c, name=f"{leg2_name} {m2} (continuous)",
                line=dict(color=AMBER, width=1.5),
                customdata=tag2, hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>"))
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

    st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) per-contract (KC/RC/CC/LCC)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Term Structure  (all active/upcoming spreads, one date)
# ══════════════════════════════════════════════════════════════════════════════

else:

    st.markdown(
        f"<div style='font-size:0.8rem;color:{MUTED};letter-spacing:0.04em;"
        f"text-transform:uppercase'>ARB Monitor &nbsp;·&nbsp; {pair_name_short}  [Term Structure]</div>",
        unsafe_allow_html=True,
    )

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
                    format_func=lambda v: "KC Z vs RC X (same year)" if v == "ZX" else "KC Z vs RC F (next year)",
                )

            n_maturities = st.slider("Number of maturities", 4, 16, 8, step=1, key="ts_n")

    st.subheader("Term Structure")

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

    st.caption("ICEBREAKER ARB  —  Data: LSEG (interim) per-contract (KC/RC/CC/LCC)")
