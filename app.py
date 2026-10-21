"""XRayLens AI: an explainable-AI laboratory for chest radiographs.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="XRayLens AI", page_icon="◐", layout="wide",
                   initial_sidebar_state="collapsed")

from xraylens.ui import inject_theme  # noqa: E402  (set_page_config must come first)

inject_theme()
page = st.navigation([
    st.Page("views/examine.py", title="Examine", default=True),
    st.Page("views/reliability.py", title="Reliability"),
    st.Page("views/provenance.py", title="Provenance"),
], position="top")
page.run()
