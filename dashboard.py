"""
S&P 500 Signal Monitor — Streamlit dashboard
============================================
Live dashboard reading from the Supabase signal_log / trade_log tables.

Run locally:   .venv/bin/streamlit run dashboard.py
Credentials come from the shared supabase_client loader (which reads .env
locally, or env vars in CI / Streamlit Cloud). No new env vars are introduced.
"""

import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="S&P 500 Signal Monitor", page_icon="\U0001F4C8",
                   layout="centered")

# Bridge Streamlit Cloud secrets into environment variables so the shared
# supabase_client loader (which reads os.getenv) works unchanged. Local runs
# rely on .env, so this is a no-op there.
try:
    for _k in ("SUPABASE_URL", "SUPABASE_KEY"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

from supabase_client import get_client  # noqa: E402  (must follow secret bridge)

# Backtest reference (hardcoded — static, ranked by realistic EV).
BACKTEST_H3_EV = 0.6933
BACKTEST_TOP5 = [
    {"signal": "H3: 20-day low (5-day hold)", "win_rate": "63.72%",
     "realistic_ev": "0.6933%", "sample": 317, "best_regime": "Bear"},
    {"signal": "H11: 20-day low + 3+ red days", "win_rate": "65.28%",
     "realistic_ev": "0.6817%", "sample": 144, "best_regime": "Bear"},
    {"signal": "H1: 5+ consecutive red days", "win_rate": "64.91%",
     "realistic_ev": "0.4084%", "sample": 57, "best_regime": "Bear"},
    {"signal": "H1: 4+ consecutive red days", "win_rate": "61.49%",
     "realistic_ev": "0.2894%", "sample": 148, "best_regime": "\u2014"},
    {"signal": "H2: single-day drop > 3%", "win_rate": "50.00%",
     "realistic_ev": "0.2619%", "sample": 46, "best_regime": "Bear"},
]


# ---------------------------------------------------------------------------
# Cached Supabase queries (refresh every 5 minutes)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_signal_log() -> pd.DataFrame:
    client = get_client()
    resp = client.table("signal_log").select("*").order("id", desc=True).execute()
    return pd.DataFrame(resp.data or [])


@st.cache_data(ttl=300)
def fetch_trade_log() -> pd.DataFrame:
    client = get_client()
    resp = client.table("trade_log").select("*").order("signal_date", desc=True).execute()
    return pd.DataFrame(resp.data or [])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def regime_badge(regime: str) -> str:
    regime = (regime or "").upper()
    colour = "#1a7f37" if regime == "BULL" else "#cf222e" if regime == "BEAR" else "#6e7781"
    label = regime if regime else "UNKNOWN"
    return (f"<span style='background:{colour};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.85rem;font-weight:600;'>{label}</span>")


def num(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001F4C8 S&P 500 Signal Monitor")
st.caption("Live dashboard of daily ^GSPC trading signals — data refreshes every 5 minutes.")

try:
    signals = fetch_signal_log()
    trades = fetch_trade_log()
    load_error = None
except Exception as exc:  # connection / credential failures
    signals = pd.DataFrame()
    trades = pd.DataFrame()
    load_error = str(exc)

if load_error:
    st.error(f"Could not connect to Supabase: {load_error[:300]}")
    st.stop()


# ---------------------------------------------------------------------------
# Section 1 — Today's Status
# ---------------------------------------------------------------------------

st.subheader("Today's Status")

if signals.empty:
    st.info("No monitor runs recorded yet. Run signal_monitor.py to populate signal_log.")
else:
    latest = signals.iloc[0]
    triggered = bool(latest.get("triggered", False))
    close = num(latest.get("spx_close"))
    change = num(latest.get("change_pct"))
    regime = (latest.get("regime") or "").upper()
    ma200 = num(latest.get("ma_200"))

    with st.container(border=True):
        if triggered:
            st.markdown("### \u26a1 SIGNAL ACTIVE")
        else:
            st.markdown("### \U0001F634 NO SIGNAL TODAY")

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "S&P 500 Close",
                f"{close:,.2f}" if close is not None else "—",
                delta=f"{change:+.2f}%" if change is not None else None,
            )
        with c2:
            st.markdown("**Regime**")
            st.markdown(regime_badge(regime), unsafe_allow_html=True)
            st.caption(f"As of {latest.get('run_date', '—')}")

        if triggered:
            st.success(
                f"**Signal:** {latest.get('signal_type', '—')}  \n"
                f"**Entry date:** {latest.get('entry_date', '—')}  \n"
                f"**Exit date:** {latest.get('exit_date', '—')}  \n"
                f"**Realistic EV per trade:** "
                f"{num(latest.get('realistic_ev_pct'), 0):.4f}%"
            )
            wr = num(latest.get("win_rate_pct"))
            if wr is not None:
                st.caption(f"Backtested win rate: {wr:.2f}%")
        else:
            st.markdown("**Next trigger levels**")
            if ma200 is not None and close is not None:
                rel = (close / ma200 - 1) * 100
                where = "above" if rel >= 0 else "below"
                st.write(f"- Close is **{abs(rel):.2f}% {where}** the 200-day MA "
                         f"({ma200:,.2f})")
            st.write("- **H3** triggers when the close prints a new 20-day low")
            st.write("- **H1 (5-day)** triggers after 5 consecutive red closes")
            st.caption("Exact next-trigger numbers are printed by signal_monitor.py each evening.")


# ---------------------------------------------------------------------------
# Section 2 — Signal History
# ---------------------------------------------------------------------------

st.subheader("Signal History")

if signals.empty:
    st.info("No signal history yet.")
else:
    hist = signals.head(30).copy()
    hist["triggered"] = hist["triggered"].apply(lambda x: "Yes" if x else "No")
    hist_view = pd.DataFrame({
        "Date": hist.get("run_date"),
        "Signal": hist.get("signal_type"),
        "Regime": hist.get("regime"),
        "Close": pd.to_numeric(hist.get("spx_close"), errors="coerce"),
        "Change %": pd.to_numeric(hist.get("change_pct"), errors="coerce"),
        "Triggered": hist.get("triggered"),
    })
    st.dataframe(
        hist_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn(width="small"),
            "Signal": st.column_config.TextColumn(width="small"),
            "Regime": st.column_config.TextColumn(width="small"),
            "Close": st.column_config.NumberColumn(format="%.2f", width="small"),
            "Change %": st.column_config.NumberColumn(format="%.2f%%", width="small"),
            "Triggered": st.column_config.TextColumn(width="small"),
        },
    )


# ---------------------------------------------------------------------------
# Section 3 — Trade Log
# ---------------------------------------------------------------------------

st.subheader("Trade Log")

if trades.empty:
    st.info("No trades logged yet — waiting for first signal.")
else:
    tl = trades.copy()
    trade_view = pd.DataFrame({
        "Signal date": tl.get("signal_date"),
        "Type": tl.get("signal_type"),
        "Regime": tl.get("regime"),
        "Entry date": tl.get("entry_date"),
        "Exit date": tl.get("exit_date"),
        "Entry": pd.to_numeric(tl.get("entry_price"), errors="coerce"),
        "Exit": pd.to_numeric(tl.get("exit_price"), errors="coerce"),
        "Return %": pd.to_numeric(tl.get("actual_return_pct"), errors="coerce"),
        "Status": tl.get("status"),
    })

    def style_row(row):
        status = str(row["Status"]).upper()
        ret = row["Return %"]
        if status == "PENDING":
            colour = "#9a6700"  # amber
        elif pd.notna(ret) and ret > 0:
            colour = "#1a7f37"  # green
        elif pd.notna(ret) and ret <= 0:
            colour = "#cf222e"  # red
        else:
            colour = "#6e7781"
        return [f"color: {colour}"] * len(row)

    styled = trade_view.style.apply(style_row, axis=1).format({
        "Entry": "{:.2f}", "Exit": "{:.2f}", "Return %": "{:+.2f}",
    }, na_rep="—")
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 4 — P&L Summary (only if at least one CLOSED trade)
# ---------------------------------------------------------------------------

closed = pd.DataFrame()
if not trades.empty and "status" in trades:
    closed = trades[trades["status"].str.upper() == "CLOSED"].copy()

if not closed.empty:
    st.subheader("P&L Summary")
    rets = pd.to_numeric(closed["actual_return_pct"], errors="coerce").dropna()
    total = len(rets)
    wins = int((rets > 0).sum())
    win_rate = 100.0 * wins / total if total else 0.0

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Trades closed", f"{total}")
        m2.metric("Win rate", f"{win_rate:.1f}%")
        m3.metric("Avg return / trade", f"{rets.mean():+.3f}%")
        m4, m5 = st.columns(2)
        m4.metric("Cumulative return", f"{rets.sum():+.2f}%")
        m5.metric("Backtested EV (H3)", f"{BACKTEST_H3_EV:.4f}%")

    chart_df = closed.copy()
    chart_df["Return %"] = pd.to_numeric(chart_df["actual_return_pct"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Return %"]).sort_values("exit_date")
    chart_df = chart_df.set_index("exit_date")[["Return %"]]
    if not chart_df.empty:
        st.bar_chart(chart_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 5 — Backtest Reference
# ---------------------------------------------------------------------------

st.subheader("Backtest Reference — Top 5 Signals by Realistic EV")
st.caption("Static reference from backtest.py (2010-present, realistic open-entry).")

ref = pd.DataFrame(BACKTEST_TOP5)
ref_view = ref.rename(columns={
    "signal": "Signal", "win_rate": "Win rate", "realistic_ev": "Realistic EV",
    "sample": "Sample", "best_regime": "Best regime",
})
st.dataframe(
    ref_view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Signal": st.column_config.TextColumn(width="medium"),
        "Win rate": st.column_config.TextColumn(width="small"),
        "Realistic EV": st.column_config.TextColumn(width="small"),
        "Sample": st.column_config.NumberColumn(width="small"),
        "Best regime": st.column_config.TextColumn(width="small"),
    },
)

st.caption("Illustrative only — not financial advice.")
