# app.py
import streamlit as st
from simulation import run_simulation, MODEL_CONFIGS

st.set_page_config(
    page_title="Misinformation Simulator",
    page_icon="📰",
    layout="wide"
)

st.title("Misinformation Spread Simulator")
st.markdown("""
This application simulates how misinformation spreads through a social network 
and the effectiveness of correction messages. Adjust parameters in the sidebar.
""")

st.sidebar.header("Simulation Controls")

seed_text = st.sidebar.text_area(
    "Seed Rumor",
    "Breaking News: Starbucks is sponsoring the Republican National Convention in Milwaukee."
)

correction_text = st.sidebar.text_area(
    "Correction Text",
    "Fact-check: False. Starbucks provided beverages to first responders in Milwaukee but did not sponsor the RNC."
)

# Add AI model selection
selected_model = st.sidebar.selectbox(
    "AI Model",
    options=list(MODEL_CONFIGS.keys()),
    index=0,
    help="Select which Gemini model to use for generating social media posts. Faster models process more quickly but may produce less nuanced text."
)

graph_size = st.sidebar.slider("Population Size", 30, 80, 50, step=5, 
                              help="Number of nodes in the social network")
T = st.sidebar.slider("Time Steps", 5, 20, 10, step=1,
                     help="Number of simulation iterations to run")
share_pct = st.sidebar.slider("Share Threshold", 0.0, 1.0, 0.5, step=0.05,
                             help="The likelihood threshold for agents to share messages (higher = less sharing)")

advanced = st.sidebar.expander("Advanced Options")
with advanced:
    show_logs = st.checkbox("Show Detailed Logs", value=False)

if st.sidebar.button("Run Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Simulating…"):
        if show_logs:
            log_container = st.expander("Simulation Logs", expanded=True)
            log_text = log_container.empty()
            logs = []
            
            def update_logs(message):
                logs.append(message)
                log_text.text("\n".join(logs[-15:]))  # Show last 15 logs
                
            # This could be implemented with a custom log handler to capture logs
            status_text.text("Running simulation...")
        
        believers = run_simulation(
            graph_size=graph_size,
            seed_text=seed_text,
            correction_text=correction_text,
            T=T,
            share_threshold=share_pct,
            model=selected_model
        )
        
        # Update progress during simulation
        for i in range(T):
            progress_bar.progress((i + 1) / T)
            status_text.text(f"Step {i+1}/{T} complete")
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.subheader("Cumulative Believers Over Time")
    
    chart_data = {"Time Step": list(range(1, T+1)), "Believers": believers}
    st.line_chart(chart_data, x="Time Step", y="Believers")
    
    final = believers[-1]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Reach", f"{final} / {graph_size}", f"{final/graph_size:.1%}")
    with col2:
        max_spread_rate = max(believers[i] - believers[i-1] for i in range(1, len(believers)))
        st.metric("Max Spread Rate", f"{max_spread_rate} per step")
    
    st.markdown(
        """
        ### Interpretation
        
        This demo simulation shows how misinformation spreads through a network and 
        how correction messages can influence the spread. The effectiveness depends on:
        
        - Trust in official sources
        - Individual susceptibility to believing claims
        - Network structure and density
        - Timing of correction messages
        
        *Note: This is a simplified model for educational purposes.*
        """
    )