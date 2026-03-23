"""
PAGE 2: SIGNALS
Bloomberg Terminal-style betting signals screen for the NBA Prop Alpha Engine.
"""

import streamlit as st
from services.ui_bridge import UIBridge
from streamlit_app.design import metric_card, badge, card_container, table_html, COLORS


def _edge_color(edge_pct: float) -> str:
    """Return the hex color for an edge percentage value."""
    if edge_pct > 3:
        return COLORS["green"]
    elif edge_pct >= 1:
        return COLORS["amber"]
    return COLORS["text_secondary"]


def _confidence_badge(level: str) -> str:
    """Return a styled badge for the confidence level."""
    level_upper = str(level).upper()
    if level_upper == "HIGH":
        return badge("HIGH", "green")
    elif level_upper == "MEDIUM" or level_upper == "MED":
        return badge("MED", "amber")
    return badge("LOW", "red")


def _action_badge(edge_pct: float) -> str:
    """Return TAKE or SKIP badge based on edge value."""
    if edge_pct > 0:
        return badge("TAKE", "green")
    return badge("SKIP", "blue")


def _build_signals_table(signals: list[dict]) -> str:
    """Build the full HTML signals table."""
    body_rows = []
    for i, s in enumerate(signals):
        bg = COLORS["surface"] if i % 2 == 0 else COLORS["bg"]
        edge_pct = float(s.get("edge_pct", 0))
        edge_col = _edge_color(edge_pct)
        conf_badge = _confidence_badge(s.get("confidence", "LOW"))
        act_badge = _action_badge(edge_pct)

        row = (
            f'<tr style="background-color:{bg};">'
            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{COLORS["text_secondary"]};'
            f'padding:8px 12px;height:40px;max-height:40px;border-bottom:1px solid {COLORS["border"]};">'
            f'{s.get("timestamp", "")}</td>'
            f'<td style="font-family:\'IBM Plex Sans\',sans-serif;font-size:11px;font-weight:500;'
            f'color:{COLORS["text_primary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("player", "")}</td>'
            f'<td style="font-family:\'IBM Plex Sans\',sans-serif;font-size:11px;'
            f'color:{COLORS["text_secondary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("event", "")}</td>'
            f'<td style="font-family:\'IBM Plex Sans\',sans-serif;font-size:11px;'
            f'color:{COLORS["text_primary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("market", "")}</td>'
            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:{COLORS["text_primary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("line", "")}</td>'
            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:{COLORS["text_primary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("model_prob", "")}</td>'
            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:{COLORS["text_primary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("market_prob", "")}</td>'
            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:600;'
            f'color:{edge_col};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{edge_pct:+.1f}%</td>'
            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:{COLORS["text_primary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("clv_expected", "")}</td>'
            f'<td style="padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{conf_badge}</td>'
            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:{COLORS["text_primary"]};padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{s.get("rec_stake", "")}</td>'
            f'<td style="padding:8px 12px;height:40px;max-height:40px;'
            f'border-bottom:1px solid {COLORS["border"]};">{act_badge}</td>'
            "</tr>"
        )
        body_rows.append(row)

    th_style = (
        f"font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:600;"
        f"color:{COLORS['text_secondary']};text-transform:uppercase;letter-spacing:0.04em;"
        f"text-align:left;padding:10px 12px;height:40px;border-bottom:1px solid {COLORS['border']};"
    )

    headers = ["Time", "Player", "Event", "Market", "Line", "Model Prob",
               "Market Prob", "Edge %", "CLV Exp", "Confidence", "Stake", "Action"]
    header_cells = "".join(f'<th style="{th_style}">{h}</th>' for h in headers)

    return (
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;border:1px solid {COLORS["border"]};'
        f'border-radius:8px;overflow:hidden;">'
        f'<thead style="background-color:{COLORS["surface_elevated"]};">'
        f'<tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table></div>'
    )


def render():
    """Render the Signals page."""

    # -- Filter Bar --------------------------------------------------------
    fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1, 1, 1, 2])

    with fc1:
        sport_filter = st.selectbox(
            "SPORT",
            ["NBA"],
            index=0,
            key="signals_sport_filter",
        )

    with fc2:
        market_filter = st.selectbox(
            "MARKET",
            ["All", "Points", "Rebounds", "Assists", "PRA", "3PM",
             "Steals", "Blocks", "Pts+Reb", "Pts+Ast", "Reb+Ast",
             "Double-Double", "Fantasy Score"],
            index=0,
            key="signals_market_filter",
        )

    with fc3:
        confidence_filter = st.selectbox(
            "CONFIDENCE",
            ["All", "HIGH", "MEDIUM", "LOW"],
            index=0,
            key="signals_confidence_filter",
        )

    with fc4:
        sort_by = st.selectbox(
            "SORT BY",
            ["Edge %", "CLV Expected", "Confidence", "Time"],
            index=0,
            key="signals_sort_by",
        )

    with fc5:
        search = st.text_input(
            "SEARCH PLAYER",
            value="",
            placeholder="Player name...",
            key="signals_search",
        )

    # -- Fetch Data --------------------------------------------------------
    signals = UIBridge.get_signals_data(
        sport_filter=sport_filter,
        confidence_filter=confidence_filter,
        sort_by=sort_by,
        search=search,
    )

    # Apply market filter client-side if not "All"
    if market_filter != "All" and signals:
        signals = [
            s for s in signals
            if str(s.get("market", "")).upper() == market_filter.upper()
        ]

    # -- Empty State -------------------------------------------------------
    if not signals:
        st.markdown(
            f'<div style="text-align:center;padding:80px 0;color:{COLORS["text_secondary"]};'
            f"font-family:'IBM Plex Sans',sans-serif;font-size:13px;\">"
            f'No signals available. Waiting for next worker run.</div>',
            unsafe_allow_html=True,
        )
        return

    # -- Summary Metrics ---------------------------------------------------
    total_signals = len(signals)
    avg_edge = sum(float(s.get("edge_pct", 0)) for s in signals) / total_signals
    high_conf_count = sum(
        1 for s in signals if str(s.get("confidence", "")).upper() == "HIGH"
    )
    total_stake = sum(float(s.get("rec_stake", 0)) for s in signals)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            metric_card("Total Signals", str(total_signals)),
            unsafe_allow_html=True,
        )
    with m2:
        edge_color = "green" if avg_edge > 2 else "amber" if avg_edge > 0 else "red"
        st.markdown(
            metric_card("Avg Edge %", f"{avg_edge:+.2f}%", delta_color=edge_color),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            metric_card("High Confidence", str(high_conf_count)),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            metric_card("Total Rec. Stake", f"{total_stake:.1f}u"),
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    # -- Signals Table -----------------------------------------------------
    st.markdown(_build_signals_table(signals), unsafe_allow_html=True)
