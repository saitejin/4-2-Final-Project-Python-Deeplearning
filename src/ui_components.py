"""
UI Components for Smart Grid Anomaly Detection
"""

import streamlit as st


def render_header():
    """Render application header"""
    st.markdown("""
        <div class="main-header">
            <h1>Smart Grid Anomaly Detection AI System</h1>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem; color: white;">
                <h3 style="margin: 0.5rem 0; color: white;">Smart Grid AI</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Navigation")
        
        if st.button("Problem Statement", key="nav_home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        if st.button("Dataset Information", key="nav_dataset", use_container_width=True):
            st.session_state.page = 'dataset'
            st.rerun()
        
        if st.button("Data Preprocessing", key="nav_preprocessing", use_container_width=True):
            st.session_state.page = 'preprocessing'
            st.rerun()
        
        if st.button("Model Training", key="nav_training", use_container_width=True):
            st.session_state.page = 'training'
            st.rerun()
        
        if st.button("Pattern Analysis", key="nav_pattern", use_container_width=True):
            st.session_state.page = 'pattern'
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### System Status")
        
        if st.session_state.model is not None:
            st.success("Model Loaded")
            st.metric("Model Type", "Conv-LSTM")
            st.metric("Accuracy", "78.92%")
            st.metric("Device", str(st.session_state.device).upper())
        else:
            st.error("Model Not Loaded")
        
        st.markdown("---")
        
        st.markdown("### Quick Stats")
        if st.session_state.data:
            data = st.session_state.data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train Samples", f"{len(data['X_train']):,}")
                st.metric("Test Samples", f"{len(data['X_test']):,}")
            with col2:
                st.metric("Features", data['X_train'].shape[1])
                st.metric("Detections", len(st.session_state.prediction_history))
