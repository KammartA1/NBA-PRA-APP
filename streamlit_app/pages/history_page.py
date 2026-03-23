"""
PAGE 4: HISTORY -- Bloomberg Terminal-style bet history with filters and export
for the NBA Prop Alpha Engine.
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from services.ui_bridge import UIBridge
from streamlit_app.design import metric_card, badge, table_html, COLORS


def _safe_str(val, default=""):
    """Return str(val) or default if val is None."""
    if val is None:
        return default
    return str(val)


def _safe_float(val, default=0.0):
    """Return float(val) or default if val is None or not numeric."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _result_badge(result):
    """Return a badge HTML string for result W/L/P/pending."""
    r = _safe_str(result).upper().strip()
    if r == "W":
        return badge("W", "green")
    elif r == "L":
        return badge("L", "red")
    elif r == "P":
        return badge("P", "amber")
    elif r == "PENDING":
        return badge("P", "amber")
    else:
        return badge(r or "-", "blue")


def _colored_number(val, fmt="{:+.2f}"):
    """Return a monospaced span colored green/red based on sign."""
    v = _safe_float(val)
    color = COLORS["green"] if v >= 0 else COLORS["red"]
    display = fmt.format(v)
    return (
        f'<span style="color:{color}; font-family: JetBrains Mono, monospace;">'
        f'{display}</span>'
    )


def _mono(val, fmt="{}"):
    """Wrap value in monospaced span."""
    return (
        f'<span style="font-family: JetBrains Mono, monospace;">'
        f'{fmt.format(val)}</span>'
    )


# NBA-specific market options for the filter dropdown
NBA_MARKET_OPTIONS = [
    "All", "Points", "Rebounds", "Assists", "PRA", "3PM",
    "Steals", "Blocks", "Pts+Reb", "Pts+Ast", "Reb+Ast",
    "Double-Double", "Fantasy Score",
]


def render():
    """Render the History page."""
    bridge = UIBridge()

    # ------------------------------------------------------------------
    # Fetch raw data
    # ------------------------------------------------------------------
    raw_data = bridge.get_history_data(filters=None) or []

    # ------------------------------------------------------------------
    # FILTER BAR
    # ------------------------------------------------------------------
    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 2])

    with f1:
        default_end = date.today()
        default_start = default_end - timedelta(days=30)
        start_date = st.date_input(
            "Start date",
            value=default_start,
            key="history_start_date",
        )
        end_date = st.date_input(
            "End date",
            value=default_end,
            key="history_end_date",
        )

    with f2:
        result_filter = st.selectbox(
            "Result",
            ["All", "Win", "Loss", "Pending"],
            key="history_result_filter",
        )

    with f3:
        market_filter = st.selectbox(
            "Market",
            NBA_MARKET_OPTIONS,
            key="history_market_filter",
        )

    with f4:
        confidence_filter = st.selectbox(
            "Confidence",
            ["All", "HIGH", "MEDIUM", "LOW"],
            key="history_confidence_filter",
        )

    with f5:
        search = st.text_input(
            "Search player / event",
            value="",
            key="history_search",
        )

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------
    filtered = list(raw_data)

    # Date filter
    if start_date and end_date:
        def _in_date_range(row):
            d = row.get("date", "")
            if not d:
                return False
            try:
                row_date = date.fromisoformat(str(d)[:10])
                return start_date <= row_date <= end_date
            except (ValueError, TypeError):
                return False
        filtered = [r for r in filtered if _in_date_range(r)]

    # Result filter
    result_map = {"Win": "W", "Loss": "L", "Pending": "PENDING"}
    if result_filter != "All":
        target = result_map.get(result_filter, result_filter)
        filtered = [
            r for r in filtered
            if _safe_str(r.get("result")).upper().strip() == target
        ]

    # Market filter
    if market_filter != "All":
        filtered = [
            r for r in filtered
            if _safe_str(r.get("market")).upper() == market_filter.upper()
        ]

    # Confidence filter
    if confidence_filter != "All":
        target_conf = confidence_filter.upper()
        filtered = [
            r for r in filtered
            if _safe_str(r.get("confidence")).upper() == target_conf
        ]

    # Search filter (case-insensitive on player or event)
    if search.strip():
        q = search.strip().lower()
        filtered = [
            r for r in filtered
            if q in _safe_str(r.get("player")).lower()
            or q in _safe_str(r.get("event")).lower()
        ]

    # Sort reverse chronological
    filtered.sort(key=lambda r: _safe_str(r.get("date"), "0000-00-00"), reverse=True)

    # ------------------------------------------------------------------
    # EXPORT CSV (top-right)
    # ------------------------------------------------------------------
    if filtered:
        df_export = pd.DataFrame(filtered)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="EXPORT CSV",
            data=csv_data,
            file_name="nba_bet_history.csv",
            mime="text/csv",
            key="history_export_csv",
        )

    # ------------------------------------------------------------------
    # MAIN TABLE
    # ------------------------------------------------------------------
    if not filtered:
        st.markdown(
            f'<p style="text-align:center; color:{COLORS["text_secondary"]}; '
            f'padding:40px 0;">No bet history yet.</p>',
            unsafe_allow_html=True,
        )
    else:
        # Build HTML table
        headers = [
            "Date", "Player", "Event", "Market", "Line", "Close",
            "CLV", "Model P", "Result", "P/L", "Stake",
        ]
        header_html = "".join(f"<th>{h}</th>" for h in headers)

        rows_html = []
        for row in filtered:
            bet_line = _safe_float(row.get("bet_line"))
            close_line = _safe_float(row.get("close_line"))
            clv = row.get("clv")
            model_prob = row.get("model_prob")
            pnl = _safe_float(row.get("pnl"))
            stake = _safe_float(row.get("stake"))

            cells = [
                f'<td>{_mono(_safe_str(row.get("date")))}</td>',
                f'<td>{_safe_str(row.get("player"))}</td>',
                f'<td>{_safe_str(row.get("event"))}</td>',
                f'<td>{_safe_str(row.get("market"))}</td>',
                f'<td>{_mono(f"{bet_line:.2f}")}</td>',
                f'<td>{_mono(f"{close_line:.2f}")}</td>',
                f'<td>{_colored_number(clv, "{:+.4f}") if clv is not None else _mono("-")}</td>',
                f'<td>{_mono(f"{_safe_float(model_prob):.1%}") if model_prob is not None else _mono("-")}</td>',
                f'<td>{_result_badge(row.get("result"))}</td>',
                f'<td>{_colored_number(pnl)}</td>',
                f'<td>{_mono(f"{stake:.2f}")}</td>',
            ]
            rows_html.append(f'<tr>{"".join(cells)}</tr>')

        table = (
            f'<table class="gqe-table">'
            f'<thead><tr>{header_html}</tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody>'
            f'</table>'
        )
        st.markdown(table, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # RUNNING TOTALS ROW
    # ------------------------------------------------------------------
    if filtered:
        total_bets = len(filtered)
        wins = sum(1 for r in filtered if _safe_str(r.get("result")).upper() == "W")
        losses = sum(1 for r in filtered if _safe_str(r.get("result")).upper() == "L")
        net_pnl = sum(_safe_float(r.get("pnl")) for r in filtered)
        total_staked = sum(_safe_float(r.get("stake")) for r in filtered)

        clv_vals = [_safe_float(r.get("clv")) for r in filtered if r.get("clv") is not None]
        avg_clv = (sum(clv_vals) / len(clv_vals)) if clv_vals else 0.0
        roi = (net_pnl / total_staked) if total_staked > 0 else 0.0

        pnl_color = "green" if net_pnl >= 0 else "red"
        clv_color = "green" if avg_clv >= 0 else "red"
        roi_color = "green" if roi >= 0 else "red"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(metric_card("Total Bets", str(total_bets)), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Wins", str(wins)), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Losses", str(losses)), unsafe_allow_html=True)
        with c4:
            st.markdown(
                metric_card("Net P/L", f"{net_pnl:+.2f}", delta_color=pnl_color),
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                metric_card("Avg CLV", f"{avg_clv:+.4f}", delta_color=clv_color),
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown(
                metric_card("ROI", f"{roi:+.2%}", delta_color=roi_color),
                unsafe_allow_html=True,
            )
