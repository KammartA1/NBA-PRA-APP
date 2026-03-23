"""
NBA Quant Engine — Refactored Entry Point
============================================
Alias for app.py. Both files use the same refactored architecture.

Usage:
    streamlit run app_refactored.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

exec(open(str(Path(__file__).resolve().parent / "app.py")).read())
