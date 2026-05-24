"""
Styles for Smart Grid Anomaly Detection Streamlit App
"""

CUSTOM_CSS = """
<style>
    /* Main theme colors - Energy/Smart Grid theme */
    :root {
        --primary-color: #00ff88;
        --secondary-color: #0a4d68;
        --success-color: #00ff88;
        --warning-color: #ffaa00;
        --danger-color: #ff3366;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0a4d68 0%, #001f3f 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 2px solid #00ff88;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0a4d68 0%, #001f3f 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
        border: 1px solid #00ff88;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #0a4d68 0%, #001f3f 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(0, 255, 136, 0.4);
        border: 2px solid #00ff88;
    }
    
    .prediction-card h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    /* Alert badges */
    .alert-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1rem;
        margin: 0.3rem;
    }
    
    .alert-safe {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: #001f3f;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        color: white;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #ff3366 0%, #cc0044 100%);
        color: white;
    }
    
    /* Info box */
    .info-box {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0a4d68 0%, #001f3f 100%);
        color: white;
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 0.75rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 255, 136, 0.4);
        border: 1px solid #00ff88;
    }
</style>
"""
