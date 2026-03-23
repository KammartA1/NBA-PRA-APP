"""
NBA Prop Alpha Engine — Bloomberg Terminal Dashboard
=====================================================
4-page institutional-grade interface.

Pages:
    1. Command Center  — system overview, hero metrics, kill switches
    2. Signals         — active betting opportunities
    3. Performance     — CLV, ROI, drawdown, calibration charts
    4. History         — full bet log with filters and export

Architecture:
    app.py (this file)
        └── streamlit_app/pages/command_center.py
        └── streamlit_app/pages/signals_page.py
        └── streamlit_app/pages/performance_page.py
        └── streamlit_app/pages/history_page.py
        └── streamlit_app/design.py              (design system)
        └── services/ui_bridge.py                (backend interface)
        └── services/*                           (computation)
        └── database/*                           (persistence)
        └── workers/*                            (background jobs)
        └── nba_engine.py                        (computation engine)

Usage:
    streamlit run app.py

Workers (run separately):
    python -m workers.scheduler
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# ── Page config (MUST be the first Streamlit call) ──────────────────────────
from streamlit_app.design import PAGE_CONFIG, GLOBAL_CSS

st.set_page_config(**PAGE_CONFIG)

# ── Initialize database ─────────────────────────────────────────────────────
from database.connection import init_db
from database.migrations import auto_migrate

try:
    init_db()
    auto_migrate()
except Exception as e:
    logging.getLogger(__name__).error("Database init failed: %s", e)

# ── Inject design system CSS ────────────────────────────────────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Navigation (visible top tabs) ──────────────────────────────────────────
from streamlit_app.pages import command_center, signals_page, performance_page, history_page

tab_cmd, tab_sig, tab_perf, tab_hist = st.tabs([
    "COMMAND CENTER",
    "SIGNALS",
    "PERFORMANCE",
    "HISTORY",
])

_tab_pages = [
    (tab_cmd, command_center),
    (tab_sig, signals_page),
    (tab_perf, performance_page),
    (tab_hist, history_page),
]

for tab, module in _tab_pages:
    with tab:
        try:
            module.render()
        except Exception as e:
            st.markdown(f"""
            <div style='padding:32px;text-align:center;'>
                <div style='font-family:JetBrains Mono,monospace;font-size:13px;
                            color:#FF4757;margin-bottom:8px;'>PAGE ERROR</div>
                <div style='font-family:IBM Plex Sans,sans-serif;font-size:11px;
                            color:#8B8B96;'>{str(e)}</div>
            </div>
            """, unsafe_allow_html=True)
            logging.getLogger(__name__).exception("Page render error")
