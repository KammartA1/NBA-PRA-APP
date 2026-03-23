"""
NBA Quant Engine v5.0 — Main Streamlit Dashboard
===================================================
Thin frontend entry point. ALL persistent state lives in the database.
ALL computation is performed in the services layer or background workers.
If Streamlit crashes and restarts, ZERO data is lost.

Architecture:
    Streamlit (this file)
        └── streamlit_app/pages/*.py  (tab renderers, read DB via services)
        └── streamlit_app/components/ (charts, sidebar)
        └── services/*               (all computation + DB read/write)
        └── database/*               (connection, ORM models, migrations)
        └── workers/*                (background ingestion, signals, retraining)

Usage:
    streamlit run app.py

Workers (run separately):
    python -m workers.scheduler        # start all workers
    python -m workers.odds_worker      # one-shot odds fetch
    python -m workers.stats_worker     # one-shot stats ingestion
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

# ── 1. Database bootstrap (idempotent — safe on every restart) ──────────────
from database.connection import init_db
from database.migrations import auto_migrate

init_db()
auto_migrate()

# ── 2. Page configuration (MUST be the first Streamlit call) ────────────────
from streamlit_app.config import setup_page, TAB_NAMES, get_user_id

setup_page()

# ── 3. Session state initialization (transient UI only) ────────────────────
from streamlit_app.state import init_session_defaults

user_id = get_user_id()
init_session_defaults(user_id)

# ── 4. Sidebar (tournament info, system status, settings — all from DB) ────
from streamlit_app.components.sidebar import render_sidebar

user_id = render_sidebar()

# ── 5. Main content area — tabs ────────────────────────────────────────────
from streamlit_app.pages import (
    model_tab,
    results_tab,
    scanner_tab,
    platforms_tab,
    history_tab,
    calibration_tab,
    insights_tab,
    alerts_tab,
    quant_tab,
    clv_tab,
    data_quality_tab,
    edge_sources_tab,
    edge_decomposition_tab,
    settings_tab,
)

tabs = st.tabs(TAB_NAMES)

with tabs[0]:
    model_tab.render()

with tabs[1]:
    results_tab.render()

with tabs[2]:
    scanner_tab.render()

with tabs[3]:
    platforms_tab.render()

with tabs[4]:
    history_tab.render()

with tabs[5]:
    calibration_tab.render()

with tabs[6]:
    insights_tab.render()

with tabs[7]:
    alerts_tab.render()

with tabs[8]:
    quant_tab.render()

with tabs[9]:
    clv_tab.render()

with tabs[10]:
    data_quality_tab.render()

with tabs[11]:
    edge_sources_tab.render()

with tabs[12]:
    edge_decomposition_tab.render()

with tabs[13]:
    settings_tab.render()

# ── 6. Footer ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:2rem;padding-top:0.8rem;border-top:1px solid #1E2D3D;
font-family:Fira Code,monospace;font-size:0.60rem;color:#2A3A4A;
display:flex;flex-wrap:wrap;gap:0.3rem 1rem;justify-content:space-between;'>
  <span>NBA QUANT ENGINE v5.0</span>
  <span style='flex:1;min-width:0;word-break:break-word;'>
    PURE FRONTEND | DB-BACKED | CRASH-SAFE | SERVICE LAYER ARCHITECTURE
  </span>
  <span>Powered by Kamal</span>
</div>
""", unsafe_allow_html=True)
