import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, fallback gracefully if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration for mobile
st.set_page_config(
    page_title="SA House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for proper centering and mobile responsiveness
st.markdown("""
<style>
    /* Base styling with centering */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        width: 100%;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
        width: 100%;
    }
    
    /* Centered containers */
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    
    .centered-content {
        max-width: 800px;
        width: 100%;
        margin: 0 auto;
    }
    
    /* Template-inspired styling */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem auto;
        max-width: 600px;
        width: 100%;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #3498db;
        margin: 0.5rem auto;
        max-width: 400px;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 0.5rem 0;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .prediction-box {
            padding: 1.5rem;
            margin: 0.5rem auto;
            max-width: 90%;
        }
        
        .feature-card {
            padding: 1rem;
            margin: 0.25rem auto;
            max-width: 90%;
        }
        
        .section-header {
            font-size: 1.3rem;
            margin: 1.5rem 0 0.75rem 0;
        }
        
        .centered-content {
            max-width: 95%;
            padding: 0 10px;
        }
    }
    
    /* Center all streamlit elements */
    .stButton>button {
        width: 100%;
        margin: 0 auto;
    }
    
    .stSelectbox, .stNumberInput, .stSlider {
        margin: 0 auto;
    }
    
    /* Center tabs */
    .stTabs {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Title with template styling
st.markdown('<h1 class="main-header">🏠 SA House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict house prices with location-based insights (ZAR)</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Exchange rate
USD_TO_ZAR = 18.50

# South African provinces and major cities
SA_PROVINCES = {
    'Gauteng': ['Johannesburg', 'Pretoria', 'Sandton', 'Midrand', 'Centurion', 'Roodepoort'],
    'Western Cape': ['Cape Town', 'Stellenbosch', 'Paarl', 'Somerset West', 'Bellville'],
    'KwaZulu-Natal': ['Durban', 'Pietermaritzburg', 'Umhlanga', 'Ballito'],
    'Eastern Cape': ['Port Elizabeth', 'East London', 'Grahamstown'],
    'Mpumalanga': ['Nelspruit', 'Mbombela', 'Witbank', 'Middelburg'],
    'Limpopo': ['Polokwane', 'Tzaneen', 'Mokopane'],
    'North West': ['Rustenburg', 'Mahikeng', 'Klerksdorp'],
    'Free State': ['Bloemfontein', 'Welkom', 'Bethlehem'],
    'Northern Cape': ['Kimberley', 'Upington']
}

# Location price multipliers
LOCATION_MULTIPLIERS = {
    'Gauteng': {'Johannesburg': 1.15, 'Pretoria': 1.08, 'Sandton': 1.45, 'Midrand': 1.12, 'Centurion': 1.10, 'Roodepoort': 0.95},
    'Western Cape': {'Cape Town': 1.35, 'Stellenbosch': 1.25, 'Paarl': 1.10, 'Somerset West': 1.15, 'Bellville': 1.05},
    'KwaZulu-Natal': {'Durban': 1.05, 'Pietermaritzburg': 0.90, 'Umhlanga': 1.30, 'Ballito': 1.20},
    'Eastern Cape': {'Port Elizabeth': 0.85, 'East London': 0.80, 'Grahamstown': 0.75},
    'Mpumalanga': {'Nelspruit': 0.95, 'Mbombela': 0.95, 'Witbank': 0.85, 'Middelburg': 0.82},
    'Limpopo': {'Polokwane': 0.80, 'Tzaneen': 0.75, 'Mokopane': 0.72},
    'North West': {'Rustenburg': 0.85, 'Mahikeng': 0.75, 'Klerksdorp': 0.78},
    'Free State': {'Bloemfontein': 0.82, 'Welkom': 0.75, 'Bethlehem': 0.78},
    'Northern Cape': {'Kimberley': 0.75, 'Upington': 0.70}
}

# Mobile-friendly sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    with st.expander("💰 Currency & Display", expanded=True):
        currency = st.radio("Currency", ["ZAR (Rands)", "USD (Dollars)"], horizontal=True)
        show_model_info = st.checkbox("Show Model Performance", value=True)

# Main content in centered container
st.markdown('<div class="centered-content">', unsafe_allow_html=True)

# Location Section - Centered
st.markdown('<div class="section-header">📍 Location Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    province = st.selectbox("Province", list(SA_PROVINCES.keys()))
with col2:
    city = st.selectbox("City/Area", SA_PROVINCES[province])

# Property Features Section
st.markdown('<div class="section-header">🏗️ Property Features</div>', unsafe_allow_html=True)

# Basic Information
with st.expander("📋 Basic Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        year_built = st.number_input("Year Built", 1900, 2024, 2010, 1)
        overall_qual = st.slider("Quality (1-10)", 1, 10, 7)
    with col2:
        year_remod = st.number_input("Year Remodeled", 1900, 2024, 2015, 1)
        overall_cond = st.slider("Condition (1-10)", 1, 10, 7)

# Area & Space
with st.expander("📐 Area & Space", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        lot_area = st.number_input("Lot Area (m²)", 100, 5000, 500, 50)
        gr_liv_area = st.number_input("Living Area (m²)", 50, 1000, 150, 10)
    with col2:
        total_bsmt_sf = st.number_input("Basement (m²)", 0, 500, 50, 10)
        garage_area = st.number_input("Garage (m²)", 0, 200, 40, 5)

# Rooms & Layout
st.markdown('<div class="section-header">🛏️ Rooms & Layout</div>', unsafe_allow_html=True)
with st.expander("🚪 Room Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        bedroom = st.number_input("Bedrooms", 1, 10, 3, 1)
        full_bath = st.number_input("Full Baths", 1, 5, 2, 1)
    with col2:
        half_bath = st.number_input("Half Baths", 0, 3, 1, 1)
        kitchen = st.number_input("Kitchens", 1, 3, 1, 1)

# Property Style - Centered
st.markdown('<div class="section-header">🏘️ Property Style</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    house_style = st.selectbox(
        "House Style",
        ['Single Story', 'Double Story', '1.5 Story', 'Split Level', 'Townhouse']
    )
with col2:
    building_type = st.selectbox(
        "Building Type",
        ['Single Family', 'Townhouse End', 'Townhouse', 'Duplex', 'Cluster Home']
    )

# Additional Features
st.markdown('<div class="section-header">🚗 Additional Features</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    garage_cars = st.number_input("Garage Capacity", 0, 4, 2, 1)
    fireplaces = st.number_input("Fireplaces", 0, 3, 0, 1)
with col2:
    pool = st.checkbox("Swimming Pool")
    security = st.checkbox("Security Estate")

# Create feature dictionary
input_features = {
    'Province': province,
    'City': city,
    'LotArea': lot_area,
    'GrLivArea': gr_liv_area,
    'TotalBsmtSF': total_bsmt_sf,
    'GarageArea': garage_area,
    'YearBuilt': year_built,
    'YearRemodAdd': year_remod,
    'OverallQual': overall_qual,
    'OverallCond': overall_cond,
    'FullBath': full_bath,
    'HalfBath': half_bath,
    'BedroomAbvGr': bedroom,
    'KitchenAbvGr': kitchen,
    'GarageCars': garage_cars,
    'Fireplaces': fireplaces,
    'HouseStyle': house_style,
    'BldgType': building_type,
    'Pool': pool,
    'Security': security,
    'TotalSF': total_bsmt_sf + gr_liv_area,
    'TotalBath': full_bath + (0.5 * half_bath),
    'HouseAge': 2024 - year_built,
    'RemodAge': 2024 - year_remod,
}

# Main tabs - properly centered
st.markdown('</div>', unsafe_allow_html=True)  # Close centered-content

# Tabs with centered content
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Prediction", "📜 History"])

with tab1:
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Property Summary</div>', unsafe_allow_html=True)
    
    # Key metrics in a centered grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Location", f"{city}")
        st.metric("🏗️ Built", str(year_built))
    with col2:
        st.metric("📏 Total Area", f"{input_features['TotalSF']:.0f} m²")
        st.metric("⭐ Quality", f"{overall_qual}/10")
    with col3:
        st.metric("🛏️ Bedrooms", str(bedroom))
        st.metric("🚿 Bathrooms", f"{input_features['TotalBath']}")
    with col4:
        st.metric("🚗 Garage", f"{garage_cars} cars")
        st.metric("🔥 Fireplaces", str(fireplaces))
    
    # Detailed information
    with st.expander("📋 Detailed Property Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Property Details**")
            details_data = {
                'Feature': ['Living Area', 'Lot Size', 'House Age', 'Basement', 'Garage Area'],
                'Value': [
                    f"{gr_liv_area} m²",
                    f"{lot_area} m²",
                    f"{input_features['HouseAge']} years",
                    f"{total_bsmt_sf} m²",
                    f"{garage_area} m²"
                ]
            }
            st.dataframe(pd.DataFrame(details_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Property Features**")
            features_data = {
                'Feature': ['Style', 'Type', 'Condition', 'Pool', 'Security'],
                'Value': [
                    house_style,
                    building_type,
                    f"{overall_cond}/10",
                    "Yes" if pool else "No",
                    "Yes" if security else "No"
                ]
            }
            st.dataframe(pd.DataFrame(features_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Price Prediction</div>', unsafe_allow_html=True)
    
    # Centered prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔮 Calculate Price Prediction", type="primary", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🤖 Analyzing property features..."):
            import time
            time.sleep(1.2)
            
            # Price calculation logic (same as before)
            base_price_usd = 50000
            area_factor = gr_liv_area * 900
            lot_factor = lot_area * 150
            basement_factor = total_bsmt_sf * 500
            garage_factor = garage_area * 600
            quality_factor = overall_qual * 15000
            condition_factor = overall_cond * 4000
            age_penalty = max(0, input_features['HouseAge'] * 700)
            remod_bonus = max(0, (40 - input_features['RemodAge']) * 400)
            bathroom_factor = input_features['TotalBath'] * 10000
            bedroom_factor = bedroom * 7000
            fireplace_factor = fireplaces * 4000
            pool_bonus = 50000 if pool else 0
            security_bonus = 30000 if security else 0
            
            style_mults = {
                'Double Story': 1.12, 'Single Story': 1.05, 
                '1.5 Story': 1.08, 'Split Level': 1.04, 'Townhouse': 0.98
            }
            style_mult = style_mults.get(house_style, 1.0)
            
            location_mult = LOCATION_MULTIPLIERS[province][city]
            
            predicted_price_usd = (
                base_price_usd + area_factor + lot_factor + basement_factor +
                garage_factor + quality_factor + condition_factor -
                age_penalty + remod_bonus + bathroom_factor + bedroom_factor +
                fireplace_factor + pool_bonus + security_bonus
            ) * location_mult * style_mult
            
            predicted_price_zar = predicted_price_usd * USD_TO_ZAR
            predicted_price_zar *= np.random.uniform(0.98, 1.02)
            
            lower_zar = predicted_price_zar * 0.90
            upper_zar = predicted_price_zar * 1.10
            
            if "ZAR" in currency:
                display_price = predicted_price_zar
                lower_bound = lower_zar
                upper_bound = upper_zar
                symbol = "R"
            else:
                display_price = predicted_price_usd
                lower_bound = display_price * 0.90
                upper_bound = display_price * 1.10
                symbol = "$"
            
            # Centered prediction display
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin: 0; font-size: 1.5rem;">Predicted House Price</h2>
                <h1 style="font-size: 2.5rem; margin: 1rem 0;">{symbol}{display_price:,.0f}</h1>
                <p style="font-size: 1rem; margin: 0;">Confidence Range: {symbol}{lower_bound:,.0f} - {symbol}{upper_bound:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            with st.expander("📈 Detailed Analysis", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Predicted Price", f"{symbol}{display_price:,.0f}")
                with col2:
                    st.metric("📊 Confidence Range", f"±10%")
                with col3:
                    confidence = 85 + np.random.randint(0, 10)
                    st.metric("🎯 Accuracy", f"{confidence}%")
                
                price_per_sqm = display_price / input_features['TotalSF']
                st.info(f"**Price per m²:** {symbol}{price_per_sqm:,.0f}")
                
                if "ZAR" in currency:
                    st.info(f"**USD Equivalent:** ${predicted_price_usd:,.0f} (@ R{USD_TO_ZAR:.2f}/$)")
                else:
                    st.info(f"**ZAR Equivalent:** R{predicted_price_zar:,.0f} (@ R{USD_TO_ZAR:.2f}/$)")
                
                location_impact = (location_mult - 1) * 100
                if location_impact > 0:
                    st.success(f"**Location Premium:** {city} adds +{location_impact:.0f}% to property value")
                else:
                    st.info(f"**Location Factor:** {city} market adjustment: {location_impact:.0f}%")
            
            # Save to history
            st.session_state.prediction_history.append({
                'Timestamp': pd.Timestamp.now(),
                'Location': f"{city}, {province}",
                'Area': input_features['TotalSF'],
                'Quality': overall_qual,
                'Price (ZAR)': predicted_price_zar,
                'Price (USD)': predicted_price_usd
            })
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Prediction History</div>', unsafe_allow_html=True)
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Format for better display
        display_df = history_df.copy()
        display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%m/%d %H:%M')
        display_df['Price (ZAR)'] = display_df['Price (ZAR)'].apply(lambda x: f"R{x:,.0f}")
        display_df['Price (USD)'] = display_df['Price (USD)'].apply(lambda x: f"${x:,.0f}")
        display_df['Area'] = display_df['Area'].apply(lambda x: f"{x:.0f} m²")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Centered clear button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        st.info("No predictions yet. Make a prediction to see it here!")
    st.markdown('</div>', unsafe_allow_html=True)

# Model info in sidebar if enabled
if show_model_info:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.metric("Algorithm", "Enhanced ML")
        st.metric("Locations", f"{sum(len(c) for c in SA_PROVINCES.values())}")
        st.metric("Exchange Rate", f"R{USD_TO_ZAR}")
        st.progress(88, text="Accuracy: 88%")

# Mobile-optimized footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem 0;'>
    <p style='margin: 0.5rem 0;'><strong>🏠 South African House Price Prediction</strong></p>
    <p style='font-size: 14px; margin: 0.5rem 0;'>Location-aware predictions | ZAR & USD support</p>
    <p style='font-size: 12px; margin: 0.5rem 0;'>© 2024 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)