import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sophisticated white theme CSS
st.markdown("""
<style>
    /* Main background - clean white */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling with elegant shadow */
    .main-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.8);
        position: relative;
        overflow: hidden;
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
        font-size: 3.5rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #6c757d;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Enhanced metric cards with subtle shadows */
    [data-testid="metric-container"] {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* Input sections with clean design */
    .stNumberInput, .stSlider {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    
    /* Premium button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1.2rem 2.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
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
    
    /* Premium result box with elegant gradient */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 3rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        border: none;
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
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .prediction-box h2 {
        color: white;
        font-size: 2.8rem;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        font-weight: 700;
        position: relative;
        z-index: 2;
    }
    
    /* Enhanced info boxes */
    .info-box {
        background: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
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
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 2rem 0;
        color: white;
    }
    
    /* Warning message styling */
    .warning-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 2rem 0;
        color: white;
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

# Premium Hero Image Section
st.markdown("""
<div style='text-align: center; margin: 3rem 0; position: relative;'>
    <div style='position: relative; border-radius: 25px; overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.15);'>
        <img src='https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=1200&h=500&fit=crop' 
             style='width: 100%; height: 400px; object-fit: cover;'
             alt='Luxury Modern House'>
        <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.9) 100%);'>
        </div>
        <div style='position: absolute; bottom: 2rem; left: 2rem; color: #2c3e50; text-align: left;'>
            <h2 style='font-size: 2.2rem; margin: 0; font-weight: 700;'>Discover Your Home's True Value</h2>
            <p style='font-size: 1.1rem; margin: 0.5rem 0 0 0; color: #6c757d; font-weight: 500;'>AI-Powered Precision Real Estate Valuation</p>
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
    st.balloons()
    st.markdown("""
    <div class="success-message">
        <h2 style='color: white; margin: 0; font-size: 2rem;'>✅ AI Model Successfully Loaded</h2>
        <p style='color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Ready for accurate property valuation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model stats in premium cards
    st.markdown("### 📊 Advanced Model Analytics")
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    tab1, tab2, tab3 = st.tabs(["🏗️ Structural Details", "📐 Spatial Analysis", "✨ Premium Features"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Quality Assessment")
            overall_qual = st.slider(
                "Overall Quality Rating",
                1, 10, 7,
                help="Architectural excellence and construction quality"
            )
            overall_cond = st.slider(
                "Current Condition",
                1, 10, 7,
                help="Present state and maintenance level"
            )
        
        with col2:
            st.markdown("#### 📅 Historical Timeline")
            year_built = st.number_input(
                "Construction Year",
                1850, 2024, 2000,
                help="Original architectural creation date"
            )
            year_remod = st.number_input(
                "Renovation Year",
                1850, 2024, 2010,
                help="Latest comprehensive modernization"
            )
            
            house_age = 2024 - year_built
            st.markdown(f"""
            <div class='info-box' style='text-align: center;'>
                <h4>🕐 Historical Context</h4>
                <p style='font-size: 1.4rem; margin: 0; color: #2c3e50;'><strong>{house_age} Years</strong> of Architectural Heritage</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Living Spaces")
            gr_liv_area = st.number_input(
                "Primary Living Area (sq ft)",
                500, 5000, 1500, 50,
                help="Above ground luxurious living space"
            )
            
            total_bsmt_sf = st.number_input(
                "Basement Excellence (sq ft)",
                0, 3000, 1000, 50,
                help="Premium below-ground square footage"
            )
            
            total_area = gr_liv_area + total_bsmt_sf
            st.markdown(f"""
            <div class='info-box' style='text-align: center;'>
                <h4>📏 Total Luxury Space</h4>
                <p style='font-size: 1.4rem; margin: 0; color: #2c3e50;'><strong>{total_area:,} sq ft</strong> of Refined Living</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🌳 Estate Dimensions")
            lot_area = st.number_input(
                "Land Estate (sq ft)",
                1000, 50000, 10000, 500,
                help="Total prestigious land holding"
            )
            
            garage_area = st.number_input(
                "Automobile Gallery (sq ft)",
                0, 1500, 400, 50,
                help="Premium vehicle accommodation space"
            )
            
            lot_acres = lot_area / 43560
            st.markdown(f"""
            <div class='info-box' style='text-align: center;'>
                <h4>🌳 Estate Scale</h4>
                <p style='font-size: 1.4rem; margin: 0; color: #2c3e50;'><strong>{lot_acres:.2f} Acres</strong> of Prime Real Estate</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🛏️ Sanctuary Suites")
            bedroom = st.number_input("Bedroom Sanctuaries", 0, 10, 3)
            st.markdown(f"<div style='text-align: center; font-size: 2rem; color: #2c3e50;'>🛏️ × {bedroom}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🚿 Spa Facilities")
            full_bath = st.number_input("Luxury Baths", 0, 5, 2)
            half_bath = st.number_input("Powder Rooms", 0, 3, 1)
            total_baths = full_bath + (0.5 * half_bath)
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 2rem; color: #2c3e50;'>🚿 × {full_bath} 🪞 × {half_bath}</div>
                <p style='font-size: 1.2rem; color: #6c757d;'>Total: {total_baths} Bathing Facilities</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### 🔥 Ambient Features")
            fireplaces = st.number_input("Elegant Fireplaces", 0, 3, 1)
            if fireplaces > 0:
                st.markdown(f"<div style='text-align: center; font-size: 2rem; color: #2c3e50;'>🔥 × {fireplaces}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center; font-size: 1.5rem; color: #adb5bd;'>❌ No Fireplace</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # PREMIUM PREDICTION BUTTON
    # ============================================
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 GENERATE INTELLIGENT VALUATION",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        
        with st.spinner("🤖 AI is conducting comprehensive property analysis..."):
            
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
                
                # Premium prediction display
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 INTELLIGENT VALUATION</h2>
                    <h1 style='font-size: 4.5rem; color: white; margin: 2rem 0; text-shadow: 3px 3px 10px rgba(0,0,0,0.3); font-weight: 800;'>
                        ${prediction:,.0f}
                    </h1>
                    <p style='font-size: 1.3rem; color: white; opacity: 0.95; font-weight: 500;'>
                        Confidence Spectrum: ${prediction * 0.9:,.0f} - ${prediction * 1.1:,.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Premium detailed metrics
                st.markdown("### 📈 Comprehensive Financial Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "💵 Market Valuation",
                        f"${prediction:,.0f}",
                        help="AI-calculated fair market value"
                    )
                
                with col2:
                    lower = prediction * 0.9
                    st.metric(
                        "📉 Conservative Estimate",
                        f"${lower:,.0f}",
                        delta=f"-10%",
                        help="Lower valuation boundary"
                    )
                
                with col3:
                    upper = prediction * 1.1
                    st.metric(
                        "📈 Premium Estimate",
                        f"${upper:,.0f}",
                        delta=f"+10%",
                        help="Upper valuation potential"
                    )
                
                with col4:
                    price_per_sqft = prediction / gr_liv_area
                    st.metric(
                        "📊 Value Density",
                        f"${price_per_sqft:.0f}",
                        help="Premium per square foot"
                    )
                
                # Premium Property Summary
                st.markdown("---")
                st.markdown("### 🏡 Executive Property Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="info-box">
                        <h4>📐 Spatial Excellence</h4>
                        <ul style='list-style: none; padding: 0; font-size: 1.1rem; color: #495057;'>
                            <li>🏠 Primary Living: <strong>{:,} sq ft</strong></li>
                            <li>🌳 Land Estate: <strong>{:,} sq ft ({:.2f} acres)</strong></li>
                            <li>🚗 Vehicle Gallery: <strong>{:,} sq ft</strong></li>
                            <li>📦 Total Domain: <strong>{:,} sq ft</strong></li>
                        </ul>
                    </div>
                    """.format(gr_liv_area, lot_area, lot_acres, garage_area, gr_liv_area + total_bsmt_sf), 
                    unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="info-box">
                        <h4>✨ Premium Attributes</h4>
                        <ul style='list-style: none; padding: 0; font-size: 1.1rem; color: #495057;'>
                            <li>⭐ Quality Excellence: <strong>{}/10</strong></li>
                            <li>🛏️ Sanctuary Suites: <strong>{}</strong></li>
                            <li>🚿 Bathing Facilities: <strong>{}</strong></li>
                            <li>🔥 Ambient Features: <strong>{}</strong></li>
                            <li>🕐 Heritage: <strong>{} years</strong></li>
                        </ul>
                    </div>
                    """.format(overall_qual, bedroom, total_baths, fireplaces, 2024 - year_built),
                    unsafe_allow_html=True)
                
                # Premium investment insights
                st.markdown("---")
                st.markdown("### 💡 Strategic Investment Intelligence")
                
                insights = []
                if overall_qual >= 8:
                    insights.append("✅ **Architectural Excellence** - Premium construction with superior materials")
                if house_age < 10:
                    insights.append("✅ **Contemporary Design** - Modern architectural standards and efficiency")
                if price_per_sqft < 150:
                    insights.append("💰 **Exceptional Value** - Competitive pricing with growth potential")
                if lot_acres > 0.5:
                    insights.append("🌳 **Estate Potential** - Significant land for future development")
                if fireplaces > 1:
                    insights.append("🔥 **Luxury Ambiance** - Multiple premium lifestyle features")
                if total_area > 3000:
                    insights.append("🏰 **Grand Residence** - Substantial living space for premium lifestyle")
                
                if insights:
                    for insight in insights:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; border-radius: 15px; margin: 1rem 0; 
                                    box-shadow: 0 8px 25px rgba(102,126,234,0.2);'>
                            <p style='color: white; font-size: 1.1rem; margin: 0; font-weight: 500;'>{insight}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("📋 **Solid Investment Foundation** - Reliable property with steady appreciation potential")
                
                # Additional luxury property image
                st.markdown("""
                <div style='text-align: center; margin: 3rem 0;'>
                    <img src='https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=800&h=400&fit=crop' 
                         style='width: 100%; max-height: 400px; object-fit: cover; border-radius: 25px; box-shadow: 0 15px 40px rgba(0,0,0,0.1);'
                         alt='Luxury Interior'>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Valuation analysis error: {e}")
                st.info("Please verify all property details are accurately entered")

else:
    # Premium model not loaded state
    st.markdown("""
    <div class="warning-message">
        <h2 style='color: white; margin: 0; font-size: 2rem;'>⚠️ AI Intelligence Pending</h2>
        <p style='color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Advanced valuation model requires initialization</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>🚀 Premium Setup Protocol</h3>
        <ol style='font-size: 1.1rem; line-height: 2.5; color: #495057;'>
            <li><strong>Save the intelligence script</strong> as <code style='background: #f8f9fa; padding: 0.3rem 0.6rem; border-radius: 6px; border: 1px solid #e9ecef;'>model_trainer_with_save.py</code></li>
            <li><strong>Ensure premium datasets</strong> <code>train.csv</code> and <code>test.csv</code> are positioned</li>
            <li><strong>Activate AI training</strong>:
                <pre style='background: #2c3e50; color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; font-size: 1.1rem;'>
python model_trainer_with_save.py</pre>
            </li>
            <li><strong>Await intelligence calibration</strong> (generates 4 premium model files)</li>
            <li><strong>Reinitialize valuation dashboard</strong></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium waiting state imagery
    st.markdown("""
    <div style='text-align: center; margin: 3rem 0;'>
        <div style='position: relative; border-radius: 25px; overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.1);'>
            <img src='https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&h=400&fit=crop' 
                 style='width: 100%; height: 400px; object-fit: cover;'
                 alt='Modern Architecture'>
            <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                        background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.9) 100%);'>
            </div>
            <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        color: #2c3e50; text-align: center;'>
                <h2 style='font-size: 2.2rem; margin: 0; font-weight: 700;'>Premium AI Valuation</h2>
                <p style='font-size: 1.2rem; margin: 1rem 0 0 0; color: #6c757d; font-weight: 500;'>Activating Intelligent Property Analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Premium footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
            border-radius: 25px; margin-top: 3rem; box-shadow: 0 10px 40px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.05);'>
    <h3 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.8rem;'>🏠 AI House Price Predictor</h3>
    <p style='font-size: 1.1rem; margin: 0; color: #6c757d;'>Advanced Machine Learning Real Estate Intelligence</p>
    <p style='font-size: 0.9rem; margin-top: 1rem; color: #adb5bd;'>© 2024 | Premium Analytics Platform | Crafted with Excellence</p>
</div>
""", unsafe_allow_html=True)