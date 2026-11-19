"""
MMCCP TEST-3 Evaluation Protocol: Standalone Interactive Viewer (eval3/ui_standalone.py).

Run directly via:
    streamlit run eval3/ui_standalone.py --server.port 8502
"""
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="TEST-3 Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from eval3.ui import render_eval_tab

def main():
    render_eval_tab()

if __name__ == "__main__":
    main()
