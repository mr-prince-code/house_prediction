import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration with mobile optimization
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive CSS for mobile devices
st.markdown("""
<style>
    /* Main background - clean white */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling with elegant shadow - Mobile responsive */
    .main-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 2rem 1rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.8);
        position: relative;
        overflow: hidden;
    }
    
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem 1rem;
            margin-bottom: 1rem;
        }
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
    }
    
    .main-header p {
        color: #6c757d;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    @media (max-width: 768px) {
        .main-header p {
            font-size: 1rem;
        }
    }
    
    /* Enhanced metric cards with subtle shadows - Mobile responsive */
    [data-testid="metric-container"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    @media (max-width: 768px) {
        [data-testid="metric-container"] {
            padding: 0.8rem;
            margin-bottom: 0.8rem;
        }
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* Input sections with clean design - Mobile responsive */
    .stNumberInput, .stSlider {
        background: #ffffff;
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    @media (max-width: 768px) {
        .stNumberInput, .stSlider {
            padding: 0.6rem;
        }
    }
    
    /* Premium button styling - Mobile responsive */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    @media (max-width: 768px) {
        .stButton button {
            font-size: 1rem;
            padding: 0.8rem 1.5rem;
        }
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton button:hover::after {
        left: 100%;
    }
    
    /* Premium result box with elegant gradient - Mobile responsive */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        border: none;
    }
    
    @media (max-width: 768px) {
        .prediction-box {
            padding: 1.5rem 1rem;
            margin: 1rem 0;
        }
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(180deg); }
    }
    
    .prediction-box h2 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        font-weight: 700;
        position: relative;
        z-index: 2;
    }
    
    @media (max-width: 768px) {
        .prediction-box h2 {
            font-size: 1.5rem;
        }
    }
    
    /* Enhanced info boxes - Mobile responsive */
    .info-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 0.8rem 0;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    @media (max-width: 768px) {
        .info-box {
            padding: 1rem;
            margin: 0.6rem 0;
        }
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .info-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Tab styling - Mobile responsive */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 12px;
    }
    
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            padding: 0.5rem;
            gap: 0.3rem;
        }
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: nowrap;
        background: #ffffff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        min-width: auto;
        flex: 1;
    }
    
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            padding: 0.6rem 0.8rem;
            font-size: 0.8rem;
        }
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
    }
    
    /* Section headers - Mobile responsive */
    .section-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    
    @media (max-width: 768px) {
        .section-header {
            padding: 1rem;
            margin: 1rem 0;
        }
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success message styling - Mobile responsive */
    .success-message {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
        color: white;
    }
    
    @media (max-width: 768px) {
        .success-message {
            padding: 1rem;
            margin: 1rem 0;
        }
    }
    
    /* Warning message styling - Mobile responsive */
    .warning-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
        color: white;
    }
    
    @media (max-width: 768px) {
        .warning-message {
            padding: 1rem;
            margin: 1rem 0;
        }
    }
    
    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        /* Reduce font sizes for mobile */
        .stMarkdown h1 {
            font-size: 1.8rem !important;
        }
        
        .stMarkdown h2 {
            font-size: 1.5rem !important;
        }
        
        .stMarkdown h3 {
            font-size: 1.3rem !important;
        }
        
        /* Adjust spacing for mobile */
        .stColumn {
            padding: 0.2rem !important;
        }
        
        /* Make images responsive */
        img {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Adjust metric values for mobile */
        [data-testid="metric-value"] {
            font-size: 1.2rem !important;
        }
        
        [data-testid="metric-label"] {
            font-size: 0.9rem !important;
        }
        
        [data-testid="metric-delta"] {
            font-size: 0.8rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header with premium design
st.markdown("""
<div class="main-header">
    <h1>🏠 AI House Price Predictor</h1>
    <p>Intelligent Property Valuation Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Premium Hero Image Section - Mobile responsive
st.markdown("""
<div style='text-align: center; margin: 2rem 0; position: relative;'>
    <div style='position: relative; border-radius: 20px; overflow: hidden; box-shadow: 0 15px 40px rgba(0,0,0,0.12);'>
        <img src='https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=1200&h=500&fit=crop' 
             style='width: 100%; height: 300px; object-fit: cover;'
             alt='Luxury Modern House'>
        <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.9) 100%);'>
        </div>
        <div style='position: absolute; bottom: 1rem; left: 1rem; color: #2c3e50; text-align: left;'>
            <h2 style='font-size: 1.5rem; margin: 0; font-weight: 700;'>Discover Your Home's True Value</h2>
            <p style='font-size: 0.9rem; margin: 0.3rem 0 0 0; color: #6c757d; font-weight: 500;'>AI-Powered Precision Real Estate Valuation</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD SAVED MODEL
# ============================================

@st.cache_resource
def load_model_files():
    """Load all saved model files"""
    try:
        if not os.path.exists('house_model.pkl'):
            return None, None, None, None
        
        with open('house_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, feature_names, metadata
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

# Load the model
model, scaler, feature_names, metadata = load_model_files()

# ============================================
# DISPLAY MODEL INFO
# ============================================

if model is not None and metadata is not None:
    
    # Success message with premium styling
    st.markdown("""
    <div class="success-message">
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'>✅ AI Model Successfully Loaded</h2>
        <p style='color: white; font-size: 1rem; margin: 0.5rem 0 0 0;'>Ready for accurate property valuation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model stats in premium cards - Mobile responsive columns
    st.markdown("### 📊 Advanced Model Analytics")
    
    # Use different column layouts for mobile vs desktop
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.metric(
            label="🤖 AI Algorithm",
            value=metadata['model_name'],
            help="Advanced machine learning architecture"
        )
    
    with col2:
        st.metric(
            label="🎯 Predictive Accuracy",
            value=f"{metadata['r2']:.1%}",
            delta="Exceptional Performance",
            help="Model explanation power for price variations"
        )
    
    with col3:
        st.metric(
            label="📉 Precision Error",
            value=f"{metadata['rmse']:.4f}",
            delta="High Precision",
            delta_color="inverse",
            help="Minimal prediction variance"
        )
    
    with col4:
        st.metric(
            label="🔢 Feature Intelligence",
            value=metadata['n_features'],
            help="Comprehensive variable analysis"
        )
    
    st.info(f"📅 **Model Training Date:** {metadata['training_date']}")
    
    st.markdown("---")
    
    # ============================================
    # PREMIUM INPUT FORM WITH TABS
    # ============================================
    
    st.markdown("### 🏡 Property Intelligence Dashboard")
    
    # Mobile-friendly tab labels
    tab1, tab2, tab3 = st.tabs(["🏗️ Structure", "📐 Space", "✨ Features"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Quality")
            overall_qual = st.slider(
                "Overall Quality Rating",
                1, 10, 7,
                help="Construction quality and excellence"
            )
            overall_cond = st.slider(
                "Current Condition",
                1, 10, 7,
                help="Present state and maintenance"
            )
        
        with col2:
            st.markdown("#### 📅 Timeline")
            year_built = st.number_input(
                "Built Year",
                1850, 2024, 2000,
                help="Original construction date"
            )
            year_remod = st.number_input(
                "Renovation Year",
                1850, 2024, 2010,
                help="Latest modernization"
            )
            
            house_age = 2024 - year_built
            st.markdown(f"""
            <div class='info-box' style='text-align: center;'>
                <h4>🕐 Property Age</h4>
                <p style='font-size: 1.2rem; margin: 0; color: #2c3e50;'><strong>{house_age} Years</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Living Area")
            gr_liv_area = st.number_input(
                "Living Area (sq ft)",
                500, 5000, 1500, 50,
                help="Above ground living space"
            )
            
            total_bsmt_sf = st.number_input(
                "Basement Area (sq ft)",
                0, 3000, 1000, 50,
                help="Below-ground square footage"
            )
            
            total_area = gr_liv_area + total_bsmt_sf
            st.markdown(f"""
            <div class='info-box' style='text-align: center;'>
                <h4>📏 Total Space</h4>
                <p style='font-size: 1.2rem; margin: 0; color: #2c3e50;'><strong>{total_area:,} sq ft</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🌳 Lot & Garage")
            lot_area = st.number_input(
                "Lot Area (sq ft)",
                1000, 50000, 10000, 500,
                help="Total land area"
            )
            
            garage_area = st.number_input(
                "Garage Area (sq ft)",
                0, 1500, 400, 50,
                help="Garage space"
            )
            
            lot_acres = lot_area / 43560
            st.markdown(f"""
            <div class='info-box' style='text-align: center;'>
                <h4>🌳 Lot Size</h4>
                <p style='font-size: 1.2rem; margin: 0; color: #2c3e50;'><strong>{lot_acres:.2f} Acres</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🛏️ Bedrooms")
            bedroom = st.number_input("Bedrooms", 0, 10, 3, key="bedroom_mobile")
            st.markdown(f"<div style='text-align: center; font-size: 1.5rem; color: #2c3e50;'>🛏️ {bedroom}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🚿 Bathrooms")
            full_bath = st.number_input("Full Baths", 0, 5, 2, key="full_bath_mobile")
            half_bath = st.number_input("Half Baths", 0, 3, 1, key="half_bath_mobile")
            total_baths = full_bath + (0.5 * half_bath)
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 1.5rem; color: #2c3e50;'>🚿 {full_bath} 🪞 {half_bath}</div>
                <p style='font-size: 1rem; color: #6c757d;'>Total: {total_baths}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### 🔥 Fireplaces")
            fireplaces = st.number_input("Fireplaces", 0, 3, 1, key="fireplaces_mobile")
            if fireplaces > 0:
                st.markdown(f"<div style='text-align: center; font-size: 1.5rem; color: #2c3e50;'>🔥 {fireplaces}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center; font-size: 1.2rem; color: #adb5bd;'>❌ None</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # PREMIUM PREDICTION BUTTON
    # ============================================
    
    predict_button = st.button(
        "🔮 GENERATE VALUATION",
        use_container_width=True,
        type="primary"
    )
    
    if predict_button:
        
        with st.spinner("🤖 Analyzing property..."):
            
            import time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'MSSubClass': [60],
                    'LotFrontage': [lot_area ** 0.5],
                    'LotArea': [lot_area],
                    'OverallQual': [overall_qual],
                    'OverallCond': [overall_cond],
                    'YearBuilt': [year_built],
                    'YearRemodAdd': [year_remod],
                    'MasVnrArea': [0],
                    'BsmtFinSF1': [total_bsmt_sf * 0.7],
                    'BsmtFinSF2': [0],
                    'BsmtUnfSF': [total_bsmt_sf * 0.3],
                    'TotalBsmtSF': [total_bsmt_sf],
                    '1stFlrSF': [gr_liv_area * 0.6],
                    '2ndFlrSF': [gr_liv_area * 0.4],
                    'LowQualFinSF': [0],
                    'GrLivArea': [gr_liv_area],
                    'BsmtFullBath': [0],
                    'BsmtHalfBath': [0],
                    'FullBath': [full_bath],
                    'HalfBath': [half_bath],
                    'BedroomAbvGr': [bedroom],
                    'KitchenAbvGr': [1],
                    'TotRmsAbvGrd': [bedroom + 2],
                    'Fireplaces': [fireplaces],
                    'GarageYrBlt': [year_built],
                    'GarageCars': [2 if garage_area > 0 else 0],
                    'GarageArea': [garage_area],
                    'WoodDeckSF': [0],
                    'OpenPorchSF': [0],
                    'EnclosedPorch': [0],
                    '3SsnPorch': [0],
                    'ScreenPorch': [0],
                    'PoolArea': [0],
                    'MiscVal': [0],
                    'MoSold': [6],
                    'YrSold': [2024],
                    'TotalSF': [total_bsmt_sf + gr_liv_area],
                    'TotalBath': [full_bath + 0.5 * half_bath],
                    'HouseAge': [2024 - year_built],
                    'RemodAge': [2024 - year_remod],
                    'HasPool': [0],
                    'HasGarage': [1 if garage_area > 0 else 0],
                    'HasBsmt': [1 if total_bsmt_sf > 0 else 0],
                    'HasFireplace': [1 if fireplaces > 0 else 0]
                })
                
                # Add dummy columns
                for feat in feature_names:
                    if feat not in input_data.columns:
                        input_data[feat] = 0
                
                input_data = input_data[feature_names]
                input_scaled = scaler.transform(input_data)
                prediction_log = model.predict(input_scaled)
                prediction = np.expm1(prediction_log[0])
                
                # Premium prediction display - Mobile responsive
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 VALUATION</h2>
                    <h1 style='font-size: 3rem; color: white; margin: 1.5rem 0; text-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-weight: 800;'>
                        ${prediction:,.0f}
                    </h1>
                    <p style='font-size: 1.1rem; color: white; opacity: 0.95; font-weight: 500;'>
                        Range: ${prediction * 0.9:,.0f} - ${prediction * 1.1:,.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Premium detailed metrics - Mobile responsive columns
                st.markdown("### 📈 Financial Analysis")
                
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                with col1:
                    st.metric(
                        "💵 Market Value",
                        f"${prediction:,.0f}",
                        help="AI-calculated fair market value"
                    )
                
                with col2:
                    lower = prediction * 0.9
                    st.metric(
                        "📉 Low Estimate",
                        f"${lower:,.0f}",
                        delta=f"-10%",
                        help="Lower valuation boundary"
                    )
                
                with col3:
                    upper = prediction * 1.1
                    st.metric(
                        "📈 High Estimate",
                        f"${upper:,.0f}",
                        delta=f"+10%",
                        help="Upper valuation potential"
                    )
                
                with col4:
                    price_per_sqft = prediction / gr_liv_area
                    st.metric(
                        "📊 Price/sq ft",
                        f"${price_per_sqft:.0f}",
                        help="Value per square foot"
                    )
                
                # Premium Property Summary
                st.markdown("---")
                st.markdown("### 🏡 Property Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="info-box">
                        <h4>📐 Space Details</h4>
                        <ul style='list-style: none; padding: 0; font-size: 1rem; color: #495057;'>
                            <li>🏠 Living: <strong>{:,} sq ft</strong></li>
                            <li>🌳 Lot: <strong>{:,} sq ft</strong></li>
                            <li>🚗 Garage: <strong>{:,} sq ft</strong></li>
                            <li>📦 Total: <strong>{:,} sq ft</strong></li>
                        </ul>
                    </div>
                    """.format(gr_liv_area, lot_area, garage_area, gr_liv_area + total_bsmt_sf), 
                    unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="info-box">
                        <h4>✨ Features</h4>
                        <ul style='list-style: none; padding: 0; font-size: 1rem; color: #495057;'>
                            <li>⭐ Quality: <strong>{}/10</strong></li>
                            <li>🛏️ Bedrooms: <strong>{}</strong></li>
                            <li>🚿 Baths: <strong>{}</strong></li>
                            <li>🔥 Fireplaces: <strong>{}</strong></li>
                            <li>🕐 Age: <strong>{} years</strong></li>
                        </ul>
                    </div>
                    """.format(overall_qual, bedroom, total_baths, fireplaces, 2024 - year_built),
                    unsafe_allow_html=True)
                
                # Premium investment insights
                st.markdown("---")
                st.markdown("### 💡 Investment Insights")
                
                insights = []
                if overall_qual >= 8:
                    insights.append("✅ **Premium Quality** - Superior construction and materials")
                if house_age < 10:
                    insights.append("✅ **Modern Property** - Contemporary design and efficiency")
                if price_per_sqft < 150:
                    insights.append("💰 **Good Value** - Competitive pricing with growth potential")
                if lot_acres > 0.5:
                    insights.append("🌳 **Large Lot** - Significant land for development")
                if fireplaces > 1:
                    insights.append("🔥 **Luxury Features** - Multiple premium amenities")
                if total_area > 3000:
                    insights.append("🏰 **Spacious** - Substantial living space")
                
                if insights:
                    for insight in insights:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1rem; border-radius: 12px; margin: 0.8rem 0; 
                                    box-shadow: 0 6px 20px rgba(102,126,234,0.2);'>
                            <p style='color: white; font-size: 1rem; margin: 0; font-weight: 500;'>{insight}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("📋 **Solid Investment** - Reliable property with steady appreciation")
                
                # Additional property image - Mobile responsive
                st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                    <img src='https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=800&h=400&fit=crop' 
                         style='width: 100%; max-height: 300px; object-fit: cover; border-radius: 20px; box-shadow: 0 12px 30px rgba(0,0,0,0.1);'
                         alt='Luxury Interior'>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Valuation error: {e}")
                st.info("Please verify all property details")

else:
    # Premium model not loaded state
    st.markdown("""
    <div class="warning-message">
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'>⚠️ AI Model Pending</h2>
        <p style='color: white; font-size: 1rem; margin: 0.5rem 0 0 0;'>Valuation model requires setup</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>🚀 Setup Instructions</h3>
        <ol style='font-size: 1rem; line-height: 2; color: #495057;'>
            <li><strong>Save script</strong> as <code style='background: #f8f9fa; padding: 0.2rem 0.4rem; border-radius: 5px; border: 1px solid #e9ecef; font-size: 0.9rem;'>model_trainer_with_save.py</code></li>
            <li><strong>Ensure datasets</strong> <code>train.csv</code> and <code>test.csv</code> are available</li>
            <li><strong>Run training</strong>:
                <pre style='background: #2c3e50; color: white; padding: 1rem; border-radius: 10px; margin: 0.8rem 0; font-size: 0.9rem; overflow-x: auto;'>
python model_trainer_with_save.py</pre>
            </li>
            <li><strong>Wait for model generation</strong> (creates 4 files)</li>
            <li><strong>Restart this app</strong></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium waiting state imagery - Mobile responsive
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <div style='position: relative; border-radius: 20px; overflow: hidden; box-shadow: 0 15px 40px rgba(0,0,0,0.1);'>
            <img src='https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&h=400&fit=crop' 
                 style='width: 100%; height: 300px; object-fit: cover;'
                 alt='Modern Architecture'>
            <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                        background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.9) 100%);'>
            </div>
            <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        color: #2c3e50; text-align: center;'>
                <h2 style='font-size: 1.5rem; margin: 0; font-weight: 700;'>AI Valuation</h2>
                <p style='font-size: 1rem; margin: 0.8rem 0 0 0; color: #6c757d; font-weight: 500;'>Activating Property Analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Premium footer - Mobile responsive
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
            border-radius: 20px; margin-top: 2rem; box-shadow: 0 8px 30px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.05);'>
    <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.3rem;'>🏠 AI House Price Predictor</h3>
    <p style='font-size: 1rem; margin: 0; color: #6c757d;'>Machine Learning Real Estate Intelligence</p>
    <p style='font-size: 0.8rem; margin-top: 1rem; color: #adb5bd;'>© 2024 | Premium Analytics Platform</p>
</div>
""", unsafe_allow_html=True)