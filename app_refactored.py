"""
NBA Prop Alpha Engine — Alias for app.py
==========================================
Usage:
    streamlit run app_refactored.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

exec(open(str(Path(__file__).resolve().parent / "app.py")).read())
