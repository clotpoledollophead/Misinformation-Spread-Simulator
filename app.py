# app.py
import streamlit as st
from simulation import run_simulation

st.set_page_config(
    page_title="Misinformation Simulator",
    page_icon="📰",
    layout="wide"
)

st.sidebar.header("Simulation Controls")

seed_text = st.sidebar.text_area(
    "Seed Rumor",
    "Breaking News: Starbucks is sponsoring the Republican National Convention in Milwaukee."
)

correction_text = st.sidebar.text_area(
    "Correction Text",
    "Fact-check: False. Starbucks provided beverages to first responders in Milwaukee but did not sponsor the RNC."
)

graph_size = st.sidebar.slider("Population Size", 30, 80, 50, step=5)
T          = st.sidebar.slider("Time Steps", 5, 20, 10, step=1)
share_pct  = st.sidebar.slider("Share Threshold", 0.0, 1.0, 0.5, step=0.05)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Simulating…"):
        believers = run_simulation(
            graph_size=graph_size,
            seed_text=seed_text,
            correction_text=correction_text,
            T=T,
            share_threshold=share_pct
        )

    st.subheader("Cumulative Believers Over Time")
    st.line_chart(believers)

    final = believers[-1]
    st.markdown(f"**Final reach:** {final} / {graph_size} ({final/graph_size:.1%})")
    st.markdown(
        "This demo simulation is based on a simplified model of misinformation spread. "
    )