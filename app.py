"""
Smart Grid Anomaly Detection System - Streamlit Application
Advanced ML-Powered Energy Theft Detection with Conv-LSTM Model
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import from src modules
from src.styles import CUSTOM_CSS
from src.model_loader import load_model
from src.ui_components import render_header, render_sidebar
from src.pages import (
    render_problem_statement_page,
    render_dataset_page,
    render_preprocessing_page,
    render_model_training_page,
    render_pattern_analysis_page
)

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Smart Grid Anomaly Detection AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    """Main application"""
    # Apply styles
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'device' not in st.session_state:
        st.session_state.device = None
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Load model on startup
    if st.session_state.model is None:
        with st.spinner("🔄 Loading AI model..."):
            model, data, device = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.data = data
                st.session_state.device = device
    
    # Render UI
    render_header()
    render_sidebar()
    
    # Route to pages
    if st.session_state.page == 'home':
        render_problem_statement_page()
    elif st.session_state.page == 'dataset':
        render_dataset_page()
    elif st.session_state.page == 'preprocessing':
        render_preprocessing_page()
    elif st.session_state.page == 'training':
        render_model_training_page()
    elif st.session_state.page == 'pattern':
        render_pattern_analysis_page()

if __name__ == "__main__":
    main()
