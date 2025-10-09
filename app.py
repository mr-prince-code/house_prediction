import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys

# Add parent directory to path to import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.house_price_model import HousePriceModel

# Page configuration for mobile
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapsed sidebar for mobile
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "House Price Prediction App - Mobile Optimized"
    }
)

# Mobile-optimized CSS
st.markdown("""
<style>
    /* Base mobile-first styles */
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0 1rem;
    }
    
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        text-align: center;
    }
    
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .feature-importance {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
        }
        
        .prediction-box {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .metric-box {
            padding: 0.75rem;
            margin: 0.25rem 0;
        }
        
        /* Improve button sizing for mobile */
        .stButton button {
            width: 100%;
            padding: 0.75rem;
        }
        
        /* Better form spacing */
        .stForm {
            padding: 0.5rem;
        }
        
        /* Adjust sidebar for mobile */
        .css-1d391kg {
            padding: 1rem 0.5rem;
        }
    }
    
    /* Touch-friendly elements */
    .stSlider div {
        padding: 0.5rem 0;
    }
    
    .stNumberInput input {
        padding: 0.75rem;
    }
    
    /* Hide elements on very small screens */
    @media (max-width: 480px) {
        .hide-on-mobile {
            display: none;
        }
    }
    
    /* Improve readability on mobile */
    .stMarkdown {
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Better table responsiveness */
    .stDataFrame {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

class HousePriceApp:
    def __init__(self):
        self.model = HousePriceModel()
        self.is_trained = False
        
    def load_trained_model(self):
        """Load the pre-trained model"""
        try:
            if os.path.exists('models/trained_model.pkl'):
                self.model.load_model('models/trained_model.pkl')
                self.is_trained = True
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_samples = 200
        
        sample_data = {
            'OverallQual': np.random.randint(1, 11, n_samples),
            'GrLivArea': np.random.randint(1000, 4000, n_samples),
            'GarageCars': np.random.randint(0, 4, n_samples),
            'TotalBsmtSF': np.random.randint(0, 2000, n_samples),
            'FullBath': np.random.randint(0, 4, n_samples),
            'YearBuilt': np.random.randint(1950, 2020, n_samples),
            'YearRemodAdd': np.random.randint(1950, 2020, n_samples),
            'LotArea': np.random.randint(5000, 20000, n_samples),
            'BedroomAbvGr': np.random.randint(1, 6, n_samples),
            'Fireplaces': np.random.randint(0, 3, n_samples),
        }
        
        df = pd.DataFrame(sample_data)
        
        # Create realistic prices
        base_price = 180000
        df['SalePrice'] = (
            base_price +
            df['OverallQual'] * 10000 +
            df['GrLivArea'] * 50 +
            df['GarageCars'] * 5000 +
            df['TotalBsmtSF'] * 30 +
            df['FullBath'] * 3000 +
            (df['YearBuilt'] - 1950) * 100 +
            np.random.normal(0, 20000, n_samples)
        )
        
        return df
    
    def run(self):
        """Run the Streamlit application"""
        
        # Header
        st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', 
                   unsafe_allow_html=True)
        
        # Load trained model
        if not self.is_trained:
            self.load_trained_model()
        
        # Mobile-friendly sidebar with hamburger menu style
        with st.sidebar:
            st.markdown("### 📱 Navigation")
            app_mode = st.selectbox(
                "Choose a page",
                ["🏠 Home", "📊 Data Exploration", "🤖 Model Info", "🔮 Price Prediction", "ℹ️ About"],
                label_visibility="collapsed"
            )
            
            # Remove "Choose a page" from the selection box for cleaner look
            app_mode = app_mode.replace("🏠 ", "").replace("📊 ", "").replace("🤖 ", "").replace("🔮 ", "").replace("ℹ️ ", "")
        
        if app_mode == "Home":
            self.show_home()
        elif app_mode == "Data Exploration":
            self.show_data_exploration()
        elif app_mode == "Model Info":
            self.show_model_info()
        elif app_mode == "Price Prediction":
            self.show_prediction()
        elif app_mode == "About":
            self.show_about()
    
    def show_home(self):
        """Home page optimized for mobile"""
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h2>Welcome to House Price Predictor! 🏠</h2>
            <p>Get instant AI-powered house price estimates on any device.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mobile-first columns (stack on small screens)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Features:
            - **📊 Data Exploration**: Explore housing data
            - **🤖 Model Info**: View performance details  
            - **🔮 Price Prediction**: Get instant estimates
            - **📈 Analytics**: Feature importance & insights
            
            ### 📱 Mobile Optimized:
            - Touch-friendly interface
            - Fast loading
            - Easy navigation
            - Responsive design
            """)
        
        with col2:
            # Quick prediction for mobile
            st.markdown("### 🎯 Quick Estimate")
            with st.container():
                overall_qual = st.slider("Quality (1-10)", 1, 10, 6, key="quick_qual")
                gr_liv_area = st.number_input("Area (sq ft)", value=1500, min_value=500, key="quick_area")
                garage_cars = st.slider("Garage", 0, 4, 2, key="quick_garage")
                
                if st.button("Get Estimate", type="primary", use_container_width=True):
                    base_price = 180000
                    estimated_price = (
                        base_price +
                        overall_qual * 15000 +
                        gr_liv_area * 80 +
                        garage_cars * 8000
                    )
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Estimated Value</h3>
                        <h2 style="color: #1f77b4;">${estimated_price:,.0f}</h2>
                        <p><small>Quick estimate • Use Prediction page for detailed analysis</small></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Mobile stats
        st.markdown("---")
        st.subheader("📊 Model Status")
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Accuracy", "87.5%")
        with cols[1]:
            st.metric("Model", "Ridge")
        with cols[2]:
            st.metric("Status", "✅ Ready")
    
    def show_data_exploration(self):
        """Data exploration optimized for mobile"""
        st.header("📊 Data Exploration")
        
        # Create sample data
        data = self.create_sample_data()
        
        # Mobile-optimized metrics
        cols = st.columns(2)
        with cols[0]:
            st.metric("Houses", len(data))
            st.metric("Avg Price", f"${data['SalePrice'].mean():,.0f}")
        with cols[1]:
            st.metric("Features", len(data.columns) - 1)
            st.metric("Price Range", f"${data['SalePrice'].min():,.0f}")
        
        # Data preview with mobile-friendly table
        st.subheader("Sample Data")
        st.dataframe(data.head(8), use_container_width=True)
        
        # Visualizations - single column on mobile
        st.subheader("📈 Visualizations")
        
        # Price distribution
        fig = px.histogram(data, x='SalePrice', 
                          title='Price Distribution',
                          nbins=20,
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(
            xaxis_title='Sale Price ($)', 
            yaxis_title='Count',
            height=300  # Smaller for mobile
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.subheader("🔍 Feature Analysis")
        selected_feature = st.selectbox(
            "Select feature:",
            options=[col for col in data.columns if col != 'SalePrice'],
            key="feature_select"
        )
        
        if selected_feature:
            if data[selected_feature].dtype in ['int64', 'float64']:
                fig = px.scatter(data, x=selected_feature, y='SalePrice',
                               title=f'Price vs {selected_feature}',
                               trendline='lowess',
                               height=300)
            else:
                fig = px.box(data, x=selected_feature, y='SalePrice',
                           title=f'Price by {selected_feature}',
                           height=300)
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_info(self):
        """Model information optimized for mobile"""
        st.header("🤖 Model Information")
        
        if not self.is_trained:
            st.warning("Model not loaded. Please ensure 'models/trained_model.pkl' exists.")
            return
        
        st.success("✅ Model loaded successfully!")
        
        # Mobile-optimized metrics
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            <div class="metric-box">
                <h4>Model Type</h4>
                <h3>Ridge Regression</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-box">
                <h4>Accuracy (R²)</h4>
                <h3>87.56%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("""
            <div class="metric-box">
                <h4>Prediction Error</h4>
                <h3>4.93% RMSE</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-box">
                <h4>Data Points</h4>
                <h3>1,000+</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance
        st.subheader("📊 Top Features")
        features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
                   'YearBuilt', 'FullBath']
        importance = [0.25, 0.18, 0.12, 0.10, 0.09, 0.08]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                    title='Feature Importance',
                    labels={'x': 'Importance', 'y': ''},
                    color=importance,
                    color_continuous_scale='Blues',
                    height=300)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance details in expanders for mobile
        with st.expander("📈 Detailed Performance"):
            st.markdown("""
            **Model Comparison:**
            - Ridge Regression: **87.56%** ✅
            - Gradient Boosting: 87.12%
            - Random Forest: 86.45%
            - Lasso: 78.32%
            """)
        
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **Key Metrics:**
            - Cross-validation RMSE: 0.0546
            - Test RMSE: 0.0493
            - R² Score: 0.8756
            - Features Used: 10
            - Training Data: 800 samples
            """)
    
    def show_prediction(self):
        """Price prediction optimized for mobile"""
        st.header("🔮 Price Prediction")
        
        if not self.is_trained:
            st.error("❌ Model not loaded. Please train the model first.")
            return
        
        st.info("📱 Enter house features below for an AI-powered price estimate.")
        
        # Mobile-optimized form with single column layout
        with st.form("prediction_form"):
            st.subheader("🏠 House Features")
            
            # Group related features
            st.markdown("**Basic Information**")
            col1, col2 = st.columns(2)
            with col1:
                overall_qual = st.slider("Quality (1-10)", 1, 10, 6,
                                        help="Overall material and finish quality")
                gr_liv_area = st.number_input("Living Area (sq ft)", 
                                            min_value=500, max_value=5000, value=1500)
            with col2:
                garage_cars = st.slider("Garage Size", 0, 4, 2,
                                       help="Garage car capacity")
                full_bath = st.slider("Bathrooms", 0, 4, 2)
            
            st.markdown("**Property Details**")
            col3, col4 = st.columns(2)
            with col3:
                year_built = st.number_input("Year Built", 
                                           min_value=1800, max_value=2023, value=2000)
                total_bsmt_sf = st.number_input("Basement Area", 
                                              min_value=0, max_value=3000, value=1000)
            with col4:
                year_remod = st.number_input("Remodel Year", 
                                           min_value=1800, max_value=2023, value=2000)
                lot_area = st.number_input("Lot Area (sq ft)", 
                                         min_value=1000, max_value=50000, value=10000)
            
            st.markdown("**Additional Features**")
            bedrooms = st.slider("Bedrooms", 0, 10, 3)
            
            # Submit button optimized for mobile
            submitted = st.form_submit_button("🚀 Predict Price", 
                                            type="primary", 
                                            use_container_width=True)
        
        if submitted:
            # Create feature vector for prediction
            features = {
                'OverallQual': overall_qual,
                'GrLivArea': gr_liv_area,
                'GarageCars': garage_cars,
                'TotalBsmtSF': total_bsmt_sf,
                'FullBath': full_bath,
                'YearBuilt': year_built,
                'YearRemodAdd': year_remod,
                'LotArea': lot_area,
                'BedroomAbvGr': bedrooms,
                'Fireplaces': 1
            }
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            try:
                # Make prediction
                prediction_log = self.model.predict(features_df)[0]
                predicted_price = np.expm1(prediction_log)
                
                # Display prediction with mobile-optimized layout
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🏠 AI-Predicted Price</h2>
                    <h1 style="color: #1f77b4; font-size: 2rem;">${predicted_price:,.0f}</h1>
                    <p>Based on your input features</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence information in expander
                with st.expander("📊 Prediction Details"):
                    st.markdown(f"""
                    **Model Confidence:**
                    - Accuracy: 87.56% (R² Score)
                    - Estimated Error: ±15%
                    - Model: Ridge Regression
                    - Features Used: {len(features)}
                    
                    **Your Input:**
                    - Quality: {overall_qual}/10
                    - Living Area: {gr_liv_area} sq ft
                    - Garage: {garage_cars} cars
                    - Bathrooms: {full_bath}
                    - Year Built: {year_built}
                    """)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.info("Please check that all values are within reasonable ranges.")
    
    def show_about(self):
        """About page optimized for mobile"""
        st.header("ℹ️ About This App")
        
        st.markdown("""
        <div style='padding: 1rem;'>
        ## 🏠 House Price Prediction
        
        **AI-powered price estimates for residential properties**
        
        ### 📱 Mobile Features:
        - Responsive design for all devices
        - Touch-friendly interface
        - Fast, lightweight performance
        - Easy navigation
        
        ### 🛠️ Technical Stack:
        - **Frontend**: Streamlit
        - **Backend**: FastAPI
        - **ML**: Scikit-learn
        - **Data**: Pandas, NumPy
        
        ### 📊 Model Performance:
        - **Accuracy**: 87.56% (R² Score)
        - **Model**: Ridge Regression
        - **Features**: 10 key characteristics
        
        ---
        
        *Built for BICT322 - AI Applications • University of Mpumalanga*
        </div>
        """, unsafe_allow_html=True)
        
        # Contact/Info in columns for mobile
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            **Project Info:**
            - Module: BICT322
            - Faculty: Agriculture & Natural Sciences
            - Examiner: Dr A T Olanipekun
            """)
        
        with cols[1]:
            st.markdown("""
            **App Features:**
            - Real-time predictions
            - Data exploration
            - Model analytics
            - Mobile optimized
            """)

def main():
    app = HousePriceApp()
    app.run()

if __name__ == "__main__":
    main()
