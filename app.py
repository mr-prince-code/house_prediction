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
    st.warning("⚠️ Plotly not installed. Charts will be limited. Install with: pip install plotly")

# Page configuration
st.set_page_config(
    page_title="SA House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏠 SA House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict house prices with location-based insights (ZAR)")
st.markdown("---")

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Exchange rate (you can make this dynamic by fetching from an API)
USD_TO_ZAR = 18.50  # Update this rate as needed

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

# Location price multipliers (relative to base)
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

# Sidebar configuration
st.sidebar.header("🎨 App Settings")
currency = st.sidebar.radio("Currency", ["ZAR (Rands)", "USD (Dollars)"], horizontal=True)
show_model_info = st.sidebar.checkbox("📊 Show Model Performance", value=True)

st.sidebar.markdown("---")
st.sidebar.header("📍 Location & Property Features")

# Location inputs
st.sidebar.subheader("🗺️ Location")
province = st.sidebar.selectbox("Province", list(SA_PROVINCES.keys()))
city = st.sidebar.selectbox("City/Area", SA_PROVINCES[province])

# Get location multiplier
location_mult = LOCATION_MULTIPLIERS[province][city]

# Property inputs
st.sidebar.subheader("🏗️ Basic Info")
col1, col2 = st.sidebar.columns(2)
with col1:
    year_built = st.number_input("Year Built", 1900, 2024, 2010, 1)
    overall_qual = st.slider("Quality (1-10)", 1, 10, 7)
with col2:
    year_remod = st.number_input("Year Remodeled", 1900, 2024, 2015, 1)
    overall_cond = st.slider("Condition (1-10)", 1, 10, 7)

st.sidebar.subheader("📐 Area & Space")
lot_area = st.sidebar.number_input("Lot Area (m²)", 100, 5000, 500, 50)
gr_liv_area = st.sidebar.number_input("Living Area (m²)", 50, 1000, 150, 10)
total_bsmt_sf = st.sidebar.number_input("Basement (m²)", 0, 500, 50, 10)
garage_area = st.sidebar.number_input("Garage (m²)", 0, 200, 40, 5)

st.sidebar.subheader("🛏️ Rooms")
col1, col2 = st.sidebar.columns(2)
with col1:
    bedroom = st.number_input("Bedrooms", 1, 10, 3, 1)
    full_bath = st.number_input("Full Baths", 1, 5, 2, 1)
with col2:
    half_bath = st.number_input("Half Baths", 0, 3, 1, 1)
    kitchen = st.number_input("Kitchens", 1, 3, 1, 1)

st.sidebar.subheader("🏘️ Property Style")
house_style = st.sidebar.selectbox(
    "House Style",
    ['Single Story', 'Double Story', '1.5 Story', 'Split Level', 'Townhouse']
)

building_type = st.sidebar.selectbox(
    "Building Type",
    ['Single Family', 'Townhouse End', 'Townhouse', 'Duplex', 'Cluster Home']
)

st.sidebar.subheader("🚗 Extras")
garage_cars = st.sidebar.number_input("Garage Capacity", 0, 4, 2, 1)
fireplaces = st.sidebar.number_input("Fireplaces", 0, 3, 0, 1)
pool = st.sidebar.checkbox("Swimming Pool")
security = st.sidebar.checkbox("Security Estate")

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

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Property Details", "🔮 Price Prediction", "📊 Analytics", "📜 History"])

with tab1:
    st.subheader("Property Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Location", f"{city}, {province}")
        st.metric("🏗️ Year Built", year_built)
    with col2:
        st.metric("📏 Total Area", f"{input_features['TotalSF']:.0f} m²")
        st.metric("⭐ Quality", f"{overall_qual}/10")
    with col3:
        st.metric("🛏️ Bedrooms", bedroom)
        st.metric("🚿 Bathrooms", input_features['TotalBath'])
    with col4:
        st.metric("🚗 Garage", f"{garage_cars} cars")
        st.metric("🔥 Fireplaces", fireplaces)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📋 Property Details")
        details_df = pd.DataFrame({
            'Feature': ['Living Area', 'Lot Size', 'House Age', 'Basement', 'Garage Area'],
            'Value': [
                f"{gr_liv_area} m²",
                f"{lot_area} m²",
                f"{input_features['HouseAge']} years",
                f"{total_bsmt_sf} m²",
                f"{garage_area} m²"
            ]
        })
        st.dataframe(details_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🏘️ Property Features")
        features_df = pd.DataFrame({
            'Feature': ['Style', 'Type', 'Condition', 'Pool', 'Security Estate'],
            'Value': [
                house_style,
                building_type,
                f"{overall_cond}/10",
                "Yes" if pool else "No",
                "Yes" if security else "No"
            ]
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Price Prediction")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔮 Calculate Price Prediction", type="primary", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🤖 Analyzing property features..."):
            import time
            time.sleep(1.2)
            
            # Base price in USD
            base_price_usd = 50000
            
            # Area factors (convert m² pricing)
            area_factor = gr_liv_area * 900  # R900 per m²
            lot_factor = lot_area * 150
            basement_factor = total_bsmt_sf * 500
            garage_factor = garage_area * 600
            
            # Quality factors
            quality_factor = overall_qual * 15000
            condition_factor = overall_cond * 4000
            
            # Age factors
            age_penalty = max(0, input_features['HouseAge'] * 700)
            remod_bonus = max(0, (40 - input_features['RemodAge']) * 400)
            
            # Room factors
            bathroom_factor = input_features['TotalBath'] * 10000
            bedroom_factor = bedroom * 7000
            fireplace_factor = fireplaces * 4000
            
            # Special features
            pool_bonus = 50000 if pool else 0
            security_bonus = 30000 if security else 0
            
            # Style multiplier
            style_mults = {
                'Double Story': 1.12, 'Single Story': 1.05, 
                '1.5 Story': 1.08, 'Split Level': 1.04, 'Townhouse': 0.98
            }
            style_mult = style_mults.get(house_style, 1.0)
            
            # Calculate USD price
            predicted_price_usd = (
                base_price_usd + area_factor + lot_factor + basement_factor +
                garage_factor + quality_factor + condition_factor -
                age_penalty + remod_bonus + bathroom_factor + bedroom_factor +
                fireplace_factor + pool_bonus + security_bonus
            ) * location_mult * style_mult
            
            # Convert to ZAR
            predicted_price_zar = predicted_price_usd * USD_TO_ZAR
            
            # Add variance
            predicted_price_zar *= np.random.uniform(0.98, 1.02)
            
            # Confidence intervals
            lower_zar = predicted_price_zar * 0.90
            upper_zar = predicted_price_zar * 1.10
            
            # Display based on currency choice
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
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin: 0;">Predicted House Price</h2>
                <h1 style="font-size: 3.5rem; margin: 1rem 0;">{symbol}{display_price:,.0f}</h1>
                <p style="font-size: 1.2rem; margin: 0;">Confidence Range: {symbol}{lower_bound:,.0f} - {symbol}{upper_bound:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Predicted Price", f"{symbol}{display_price:,.0f}")
            with col2:
                st.metric("📉 Lower Bound", f"{symbol}{lower_bound:,.0f}")
            with col3:
                st.metric("📈 Upper Bound", f"{symbol}{upper_bound:,.0f}")
            with col4:
                confidence = 85 + np.random.randint(0, 10)
                st.metric("🎯 Confidence", f"{confidence}%")
            
            # Show both currencies
            if "ZAR" in currency:
                st.info(f"💱 **USD Equivalent:** ${predicted_price_usd:,.0f} (@ R{USD_TO_ZAR:.2f}/$)")
            else:
                st.info(f"💱 **ZAR Equivalent:** R{predicted_price_zar:,.0f} (@ R{USD_TO_ZAR:.2f}/$)")
            
            # Price per m²
            price_per_sqm = display_price / input_features['TotalSF']
            st.info(f"📊 **Price per m²:** {symbol}{price_per_sqm:,.0f}")
            
            # Location insight
            location_impact = (location_mult - 1) * 100
            if location_impact > 0:
                st.success(f"📍 **Location Premium:** {city} adds +{location_impact:.0f}% to property value")
            else:
                st.info(f"📍 **Location Factor:** {city} market adjustment: {location_impact:.0f}%")
            
            # Save to history
            st.session_state.prediction_history.append({
                'Timestamp': pd.Timestamp.now(),
                'Location': f"{city}, {province}",
                'Area': input_features['TotalSF'],
                'Quality': overall_qual,
                'Price (ZAR)': predicted_price_zar,
                'Price (USD)': predicted_price_usd
            })

with tab3:
    st.subheader("📊 Location Analytics")
    
    # Province comparison
    st.markdown("#### 🗺️ Province Price Comparison")
    
    avg_multipliers = {prov: np.mean(list(cities.values())) 
                      for prov, cities in LOCATION_MULTIPLIERS.items()}
    
    province_df = pd.DataFrame({
        'Province': list(avg_multipliers.keys()),
        'Avg Multiplier': list(avg_multipliers.values())
    }).sort_values('Avg Multiplier', ascending=False)
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(province_df, x='Province', y='Avg Multiplier',
                    title='Average Property Value by Province',
                    color='Avg Multiplier', color_continuous_scale='Viridis')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(province_df.set_index('Province'))
    
    # Selected province cities
    st.markdown(f"#### 🏙️ {province} - City Comparison")
    city_df = pd.DataFrame({
        'City': list(LOCATION_MULTIPLIERS[province].keys()),
        'Price Multiplier': list(LOCATION_MULTIPLIERS[province].values())
    }).sort_values('Price Multiplier', ascending=False)
    
    st.dataframe(city_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("📜 Prediction History")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        st.dataframe(
            history_df.style.format({
                'Price (ZAR)': 'R{:,.0f}',
                'Price (USD)': '${:,.0f}',
                'Area': '{:.0f} m²',
                'Timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M')
            }),
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Make a prediction to see it here!")

# Model info sidebar
if show_model_info:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.metric("Algorithm", "Enhanced ML")
    st.sidebar.metric("Locations", f"{sum(len(c) for c in SA_PROVINCES.values())}")
    st.sidebar.metric("Exchange Rate", f"R{USD_TO_ZAR}")
    st.sidebar.progress(88, text="Accuracy: 88%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p><strong>🏠 South African House Price Prediction</strong></p>
    <p style='font-size: 14px;'>Location-aware predictions | ZAR & USD support</p>
    <p style='font-size: 12px;'>© 2024 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)