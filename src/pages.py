"""
Page Rendering Functions for Smart Grid Anomaly Detection
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import time
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report


def render_problem_statement_page():
    """Render problem statement page (Home)"""
    
    # Display smart grid image
    try:
        grid_img = Image.open("Gemini_Generated_Image_rzgsxmrzgsxmrzgs.png")
        st.image(grid_img, width=500)
    except:
        pass
    
    st.markdown("""
    ### **Problem Statement**  
    Smart grids are modern electrical networks that use digital technology to monitor and manage the transport 
    of electricity from generation sources to consumers. However, these advanced systems face critical security 
    challenges that threaten their reliability and efficiency:  

    **Energy Theft** – Unauthorized consumption of electricity through meter tampering or bypassing, causing 
    significant financial losses to utility companies (estimated at billions of dollars annually).  
    **False Data Injection (FDI) Attacks** – Malicious actors manipulate sensor readings and consumption data, 
    leading to incorrect billing, grid instability, and potential blackouts.  
    **Real-Time Detection Challenges** – Traditional rule-based systems struggle to identify sophisticated 
    attack patterns hidden in massive volumes of consumption data.  
    **High-Stakes Impact** – Undetected anomalies can cascade into grid failures, affecting millions of 
    consumers and critical infrastructure.  

    ---

    #### **Project Objective**  
    This research aims to develop an **AI-powered smart grid anomaly detection system** that leverages 
    **deep learning and temporal pattern recognition** to accurately detect energy theft and false data 
    injection attacks in real-time. The proposed approach includes:  

    **Conv-LSTM Hybrid Architecture** – Combines Convolutional Neural Networks (CNN) for spatial feature 
    extraction with Long Short-Term Memory (LSTM) networks for temporal pattern analysis.  
    **Temporal Pattern Recognition** – Analyzes time-series energy consumption data to identify abnormal 
    usage patterns that indicate attacks or theft.  
    **High Accuracy Detection** – Achieves **78.92% accuracy** with **82.08% recall** for attack detection, 
    minimizing false negatives in critical security scenarios.  
    **Balanced Training with SMOTE** – Addresses class imbalance in attack vs. normal consumption data to 
    ensure robust performance across both categories.  
    **Robust Preprocessing Pipeline** – Implements winsorization, temporal smoothing, and standard scaling 
    to handle noisy real-world smart meter data.  

    This approach enhances the security and reliability of smart grid infrastructure, enabling utility companies 
    to **detect threats faster, reduce financial losses, and maintain grid stability** for millions of consumers.  

    ---

    #### **Deep Learning Model Architecture**  
    The proposed system incorporates advanced deep learning techniques:  

    **Conv-LSTM Hybrid Model** – Our proposed architecture achieving **78.92% accuracy** on energy consumption data.  
    **Convolutional Layer** – Extracts spatial features from consumption patterns (128 channels, kernel size 5).  
    **LSTM Layer** – Captures temporal dependencies in time-series data (128 hidden units, 2 layers).  
    **Fully Connected Layers** – Final classification with 512-neuron dense layer and binary output.  
    **PyTorch Implementation** – Leverages GPU acceleration for efficient training and real-time inference.  

    ---
    """)


def render_dataset_page():
    """Render dataset information page"""
    st.title("Dataset Information")
    
    if st.session_state.data is None:
        st.error("Data not loaded. Please restart the application.")
        return
    
    data = st.session_state.data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    st.markdown("""
    <div class="info-box">
        <h3>Smart Grid Energy Consumption Dataset</h3>
        <p>
            This dataset contains time-series energy consumption patterns from smart meters, including both 
            normal usage and anomalous patterns (energy theft and false data injection attacks).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(X_train):,}</h3>
            <p>Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(X_test):,}</h3>
            <p>Testing Samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{X_train.shape[1]}</h3>
            <p>Time Steps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>2</h3>
            <p>Classes</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_normal = np.sum(y_train == 0)
        train_attack = np.sum(y_train == 1)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Normal', 'Attack'],
                y=[train_normal, train_attack],
                text=[f'{train_normal:,}', f'{train_attack:,}'],
                textposition='auto',
                marker_color=['#00ff88', '#ff3366']
            )
        ])
        
        fig.update_layout(
            title="Training Set Distribution",
            yaxis_title="Count",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        test_normal = np.sum(y_test == 0)
        test_attack = np.sum(y_test == 1)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Normal', 'Attack'],
                y=[test_normal, test_attack],
                text=[f'{test_normal:,}', f'{test_attack:,}'],
                textposition='auto',
                marker_color=['#00ff88', '#ff3366']
            )
        ])
        
        fig.update_layout(
            title="Testing Set Distribution",
            yaxis_title="Count",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Dataset Details")
    
    dataset_info = {
        'Property': [
            'Total Samples',
            'Training Samples',
            'Testing Samples',
            'Time Steps per Sample',
            'Features per Time Step',
            'Classes',
            'Data Type',
            'Preprocessing',
            'Balancing Method',
            'Train/Test Split'
        ],
        'Value': [
            f'{len(X_train) + len(X_test):,}',
            f'{len(X_train):,}',
            f'{len(X_test):,}',
            f'{X_train.shape[1]}',
            f'{X_train.shape[2]}',
            'Binary (Normal: 0, Attack: 1)',
            'Time-series energy consumption',
            'Winsorization, Smoothing, Standard Scaling',
            'SMOTE (Synthetic Minority Over-sampling)',
            '80% Train / 20% Test'
        ]
    }
    
    df_info = pd.DataFrame(dataset_info)
    st.dataframe(df_info, use_container_width=True, hide_index=True)
    
    st.markdown("### 🔍 Sample Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Normal Consumption Pattern")
        normal_indices = np.where(y_test == 0)[0]
        if len(normal_indices) > 0:
            sample_idx = np.random.choice(normal_indices)
            sample = X_test[sample_idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(sample))),
                y=sample[:, 0],
                mode='lines',
                name='Energy Reading',
                line=dict(color='#00ff88', width=2)
            ))
            fig.update_layout(
                xaxis_title="Time Step",
                yaxis_title="Normalized Energy",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Attack Pattern")
        attack_indices = np.where(y_test == 1)[0]
        if len(attack_indices) > 0:
            sample_idx = np.random.choice(attack_indices)
            sample = X_test[sample_idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(sample))),
                y=sample[:, 0],
                mode='lines',
                name='Energy Reading',
                line=dict(color='#ff3366', width=2)
            ))
            fig.update_layout(
                xaxis_title="Time Step",
                yaxis_title="Normalized Energy",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Set Statistics")
        train_stats = {
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
            'Value': [
                f'{np.mean(X_train):.4f}',
                f'{np.std(X_train):.4f}',
                f'{np.min(X_train):.4f}',
                f'{np.max(X_train):.4f}',
                f'{np.median(X_train):.4f}'
            ]
        }
        st.dataframe(pd.DataFrame(train_stats), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Testing Set Statistics")
        test_stats = {
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
            'Value': [
                f'{np.mean(X_test):.4f}',
                f'{np.std(X_test):.4f}',
                f'{np.min(X_test):.4f}',
                f'{np.max(X_test):.4f}',
                f'{np.median(X_test):.4f}'
            ]
        }
        st.dataframe(pd.DataFrame(test_stats), use_container_width=True, hide_index=True)


def render_pattern_analysis_page():
    """Render consumption pattern analysis page"""
    st.title("Pattern Analysis")
    
    if st.session_state.model is None or st.session_state.data is None:
        st.error("Model or data not loaded. Please restart the application.")
        return
    
    st.markdown("""
    <div class="info-box">
        <p>
            Select an energy consumption pattern from <b>test data</b> (unseen by model during training) 
            to get real-time anomaly detection with threat assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Random Sample")
    
    if st.button("New Sample", use_container_width=True):
        st.rerun()
    
    data = st.session_state.data
    X_test = data['X_test']
    y_test = data['y_test']
    
    sample_idx = np.random.randint(0, len(X_test))
    
    sample_data = X_test[sample_idx:sample_idx+1]
    true_label = y_test[sample_idx]
    
    st.success(f"Loaded Test Pattern #{sample_idx} | True Label: {'Attack' if true_label == 1 else 'Normal'}")
    
    with st.expander("View Consumption Pattern (First 20 Time Steps)"):
        feature_df = pd.DataFrame({
            'Time Step': [f'T_{i+1}' for i in range(min(20, sample_data.shape[1]))],
            'Energy Reading': sample_data[0, :20, 0]
        })
        st.dataframe(feature_df, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(sample_data.shape[1])),
            y=sample_data[0, :, 0],
            mode='lines',
            name='Energy Consumption',
            line=dict(color='#00ff88', width=2)
        ))
        fig.update_layout(
            title="Energy Consumption Pattern Over Time",
            xaxis_title="Time Step",
            yaxis_title="Normalized Energy Reading",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("Analyze Pattern", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = st.session_state.model
        device = st.session_state.device
        
        status_text.text("🔍 Analyzing consumption pattern...")
        progress_bar.progress(33)
        time.sleep(0.5)
        
        with torch.no_grad():
            sample_tensor = torch.from_numpy(sample_data).float().permute(0, 2, 1).to(device)
            logits = model(sample_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        prediction = (probs > 0.5).astype(int)[0]
        attack_prob = probs[0]
        
        status_text.text("Assessing threat level...")
        progress_bar.progress(66)
        time.sleep(0.5)
        
        if attack_prob < 0.3:
            threat_level = "LOW"
            threat_color = "alert-safe"
            action = "ALLOW"
        elif attack_prob < 0.7:
            threat_level = "MODERATE"
            threat_color = "alert-warning"
            action = "MONITOR"
        else:
            threat_level = "HIGH"
            threat_color = "alert-danger"
            action = "BLOCK"
        
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success("Analysis Complete!")
        
        st.markdown("### Detection Result")
        
        st.markdown(f"""
        <div class="prediction-card">
            <h2>{"ANOMALY DETECTED" if prediction == 1 else "NORMAL CONSUMPTION"}</h2>
            <p style="font-size: 1.2rem; margin-top: 1rem;">
                Attack Probability: {attack_prob:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{threat_level}</h3>
                <p>Threat Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{action}</h3>
                <p>Recommended Action</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            is_correct = (prediction == true_label)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{"CORRECT" if is_correct else "INCORRECT"}</h3>
                <p>vs True Label</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Probability Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prob_df = pd.DataFrame({
                'Class': ['Normal', 'Attack'],
                'Probability': [1 - attack_prob, attack_prob]
            })
            
            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df['Class'],
                    y=prob_df['Probability'],
                    text=[f'{p:.1%}' for p in prob_df['Probability']],
                    textposition='auto',
                    marker_color=['#00ff88', '#ff3366']
                )
            ])
            
            fig.update_layout(
                title="Classification Probabilities",
                yaxis_title="Probability",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = attack_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Attack Probability %"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#0a4d68"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4edda"},
                        {'range': [30, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.prediction_history.append({
            'sample_idx': int(sample_idx),
            'prediction': int(prediction),
            'probability': float(attack_prob),
            'threat_level': threat_level,
            'action': action,
            'correct': bool(is_correct)
        })


def render_preprocessing_page():
    """Render data preprocessing page showing raw data, steps, and preprocessed data"""
    st.title("Data Preprocessing Pipeline")
    
    st.markdown("""
    <div class="info-box">
        <h3>Data Preprocessing Steps</h3>
        <p>
            This page demonstrates the preprocessing pipeline applied to raw smart grid consumption data.
            The process includes cleaning, outlier removal, smoothing, scaling, and class balancing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.error("Data not loaded. Please restart the application.")
        return
    
    # Load raw data
    try:
        import pandas as pd
        raw_df = pd.read_csv("data/datasetsmall.csv")
        
        st.markdown("### Step 1: Raw Data")
        st.write(f"**Shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(raw_df):,}")
        with col2:
            st.metric("Features", len([c for c in raw_df.columns if c not in ['CONS_NO', 'FLAG']]))
        with col3:
            if 'FLAG' in raw_df.columns:
                normal_count = (raw_df['FLAG'] == 0).sum()
                attack_count = (raw_df['FLAG'] == 1).sum()
                st.metric("Normal/Attack", f"{normal_count}/{attack_count}")
        
        with st.expander("View Raw Data Sample"):
            st.dataframe(raw_df.head(20), use_container_width=True)
        
        # Class distribution
        st.markdown("### Step 2: Class Distribution (Before Balancing)")
        
        if 'FLAG' in raw_df.columns:
            normal_count = (raw_df['FLAG'] == 0).sum()
            attack_count = (raw_df['FLAG'] == 1).sum()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Normal', 'Attack'],
                    y=[normal_count, attack_count],
                    text=[f'{normal_count:,}<br>({normal_count/len(raw_df)*100:.1f}%)', 
                          f'{attack_count:,}<br>({attack_count/len(raw_df)*100:.1f}%)'],
                    textposition='auto',
                    marker_color=['#00ff88', '#ff3366']
                )
            ])
            fig.update_layout(
                title="Original Class Distribution (Imbalanced)",
                yaxis_title="Count",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Preprocessing steps
        st.markdown("### Step 3: Preprocessing Pipeline")
        
        steps_data = {
            'Step': [
                '1. Data Cleaning',
                '2. Outlier Removal',
                '3. Temporal Smoothing',
                '4. Standard Scaling',
                '5. SMOTE Balancing',
                '6. Train/Test Split'
            ],
            'Description': [
                'Convert to numeric, interpolate missing values',
                'Winsorization (0.5% - 99.5% quantiles)',
                '3-point moving average smoothing',
                'Standardize features (mean=0, std=1)',
                'Oversample minority class to balance dataset',
                '80% training, 20% testing (stratified)'
            ],
            'Output': [
                'Clean numeric data',
                'Outliers clipped',
                'Smoothed temporal patterns',
                'Normalized features',
                'Balanced classes',
                'Train/Test sets'
            ]
        }
        
        steps_df = pd.DataFrame(steps_data)
        st.dataframe(steps_df, use_container_width=True, hide_index=True)
        
        # Visualize preprocessing effects
        st.markdown("### Step 4: Preprocessing Effects Visualization")
        
        feature_cols = [col for col in raw_df.columns if col not in {'CONS_NO', 'FLAG'}]
        sample_idx = 0
        
        # Get raw sample
        raw_sample = raw_df[feature_cols].iloc[sample_idx].values
        
        # Show before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Before Preprocessing")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=raw_sample[:50],
                mode='lines+markers',
                name='Raw Data',
                line=dict(color='#ff3366', width=2),
                marker=dict(size=4)
            ))
            fig.update_layout(
                xaxis_title="Time Step",
                yaxis_title="Energy Reading",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Mean:** {np.mean(raw_sample):.2f}")
            st.write(f"**Std Dev:** {np.std(raw_sample):.2f}")
            st.write(f"**Min:** {np.min(raw_sample):.2f}")
            st.write(f"**Max:** {np.max(raw_sample):.2f}")
        
        with col2:
            st.markdown("#### After Preprocessing")
            # Get preprocessed sample from loaded data
            data = st.session_state.data
            preprocessed_sample = data['X_train'][sample_idx, :, 0]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=preprocessed_sample[:50],
                mode='lines+markers',
                name='Preprocessed Data',
                line=dict(color='#00ff88', width=2),
                marker=dict(size=4)
            ))
            fig.update_layout(
                xaxis_title="Time Step",
                yaxis_title="Normalized Energy",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Mean:** {np.mean(preprocessed_sample):.2f}")
            st.write(f"**Std Dev:** {np.std(preprocessed_sample):.2f}")
            st.write(f"**Min:** {np.min(preprocessed_sample):.2f}")
            st.write(f"**Max:** {np.max(preprocessed_sample):.2f}")
        
        # Final dataset stats
        st.markdown("### Step 5: Final Preprocessed Dataset")
        
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(data['X_train']):,}</h3>
                <p>Training Samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(data['X_test']):,}</h3>
                <p>Testing Samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            train_normal = np.sum(data['y_train'] == 0)
            train_attack = np.sum(data['y_train'] == 1)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{train_normal:,}/{train_attack:,}</h3>
                <p>Train Normal/Attack</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            test_normal = np.sum(data['y_test'] == 0)
            test_attack = np.sum(data['y_test'] == 1)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{test_normal:,}/{test_attack:,}</h3>
                <p>Test Normal/Attack</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Balanced class distribution
        st.markdown("### Step 6: Class Distribution (After SMOTE Balancing)")
        
        train_normal = np.sum(data['y_train'] == 0)
        train_attack = np.sum(data['y_train'] == 1)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Normal', 'Attack'],
                y=[train_normal, train_attack],
                text=[f'{train_normal:,}<br>({train_normal/(train_normal+train_attack)*100:.1f}%)', 
                      f'{train_attack:,}<br>({train_attack/(train_normal+train_attack)*100:.1f}%)'],
                textposition='auto',
                marker_color=['#00ff88', '#ff3366']
            )
        ])
        fig.update_layout(
            title="Balanced Training Set Distribution",
            yaxis_title="Count",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("⚠️ Raw data file not found. Showing preprocessed data only.")
        
        data = st.session_state.data
        st.markdown("### Preprocessed Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", f"{len(data['X_train']):,}")
            st.metric("Testing Samples", f"{len(data['X_test']):,}")
        with col2:
            st.metric("Time Steps", data['X_train'].shape[1])
            st.metric("Features", data['X_train'].shape[2])


def render_model_training_page():
    """Render model training page with graphs and visualizations"""
    st.title("Model Training & Performance")
    
    st.markdown("""
    <div class="info-box">
        <h3>Conv-LSTM Model Performance Analysis</h3>
        <p>
            Comprehensive performance metrics, visualizations, and analysis of the Conv-LSTM hybrid model
            for smart grid anomaly detection.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Overview
    st.markdown("### Model Architecture Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Conv-LSTM Hybrid Architecture**
        
        The model combines Convolutional Neural Networks (CNN) for spatial feature extraction with 
        Long Short-Term Memory (LSTM) networks for temporal pattern analysis.
        
        **Architecture Layers:**
        - **Convolutional Layer:** 1 → 128 channels, kernel size 5, ReLU, BatchNorm, MaxPool, Dropout(0.3)
        - **LSTM Layer:** 128 input features, 128 hidden units, 2 layers, Dropout(0.3)
        - **Fully Connected:** 128 → 512 → 1 (Binary output)
        
        **Training Configuration:**
        - Optimizer: Adam (lr=5e-4, weight decay=1e-5)
        - Loss: Binary Cross Entropy
        - Max Epochs: 250 with early stopping (patience=20)
        - Gradient Clipping: 1.0
        """)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>78.92%</h3>
            <p>Overall Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>82.08%</h3>
            <p>Attack Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown("### Performance Metrics")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (Class 0)', 'Precision (Class 1)', 'Recall (Class 0)', 'Recall (Class 1)', 'F1-Score (Class 0)', 'F1-Score (Class 1)', 'Macro Avg F1'],
        'Score': [78.92, 80.87, 77.18, 75.73, 82.08, 78.21, 79.58, 78.90]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Metrics Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance Bar Chart")
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_data['Metric'][:4],
                y=metrics_data['Score'][:4],
                text=[f'{s:.2f}%' for s in metrics_data['Score'][:4]],
                textposition='auto',
                marker_color=['#00ff88', '#00d4ff', '#ffaa00', '#ff3366']
            )
        ])
        fig.update_layout(
            yaxis_title="Score (%)",
            yaxis_range=[70, 85],
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Performance Radar Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[78.92, 80.87, 82.08, 78.90],
            theta=['Accuracy', 'Precision (0)', 'Recall (1)', 'F1-Score'],
            fill='toself',
            name='Conv-LSTM',
            line=dict(color='#00ff88', width=2)
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[70, 90])),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix Analysis")
    
    cm = np.array([[3597, 1152], [851, 3897]])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Attack'],
            y=['Actual Normal', 'Actual Attack'],
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 20},
            colorscale='Greens',
            showscale=True
        ))
        fig.update_layout(
            title="Confusion Matrix - Test Set",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Confusion Matrix Breakdown:**")
        st.write(f"**True Negatives:** {cm[0,0]:,}")
        st.write(f"   (Normal correctly identified)")
        st.write(f"**False Positives:** {cm[0,1]:,}")
        st.write(f"   (Normal misclassified as Attack)")
        st.write(f"**False Negatives:** {cm[1,0]:,}")
        st.write(f"   (Attack misclassified as Normal)")
        st.write(f"**True Positives:** {cm[1,1]:,}")
        st.write(f"   (Attack correctly identified)")
        
        total = cm.sum()
        correct = cm[0,0] + cm[1,1]
        st.write(f"\n**Total Predictions:** {total:,}")
        st.write(f"**Correct:** {correct:,} ({correct/total*100:.2f}%)")
    
    # Class-wise Performance
    st.markdown("### Class-wise Performance Comparison")
    
    class_metrics = {
        'Class': ['Normal (0)', 'Attack (1)'],
        'Precision': [80.87, 77.18],
        'Recall': [75.73, 82.08],
        'F1-Score': [78.21, 79.58],
        'Support': [4749, 4748]
    }
    
    class_df = pd.DataFrame(class_metrics)
    st.dataframe(class_df, use_container_width=True, hide_index=True)
    
    # Grouped bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Precision',
        x=class_metrics['Class'],
        y=class_metrics['Precision'],
        text=[f'{p:.2f}%' for p in class_metrics['Precision']],
        textposition='auto',
        marker_color='#00ff88'
    ))
    fig.add_trace(go.Bar(
        name='Recall',
        x=class_metrics['Class'],
        y=class_metrics['Recall'],
        text=[f'{r:.2f}%' for r in class_metrics['Recall']],
        textposition='auto',
        marker_color='#00d4ff'
    ))
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=class_metrics['Class'],
        y=class_metrics['F1-Score'],
        text=[f'{f:.2f}%' for f in class_metrics['F1-Score']],
        textposition='auto',
        marker_color='#ffaa00'
    ))
    fig.update_layout(
        title="Class-wise Performance Metrics",
        yaxis_title="Score (%)",
        yaxis_range=[70, 90],
        barmode='group',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Insights
    st.markdown("### Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Strengths:**
        - High attack recall (82.08%) - Good at detecting threats
        - Balanced performance across both classes
        - Strong precision for normal consumption (80.87%)
        - Temporal pattern recognition via LSTM
        """)
    
    with col2:
        st.markdown("""
        **Areas for Improvement:**
        - Increase dataset size (currently using subset)
        - Hyperparameter optimization with Optuna
        - Target accuracy: 91.3% (paper benchmark)
        - Reduce false positives for normal class
        """)
