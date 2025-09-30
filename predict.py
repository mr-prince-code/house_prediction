import joblib
import pandas as pd
import numpy as np

def load_house_price_model():
    """Load the saved model package"""
    try:
        model_package = joblib.load('sa_house_price_model.joblib')
        print("✅ Model loaded successfully!")
        return model_package
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return None

def predict_house_price(model_package, input_data):
    """Make prediction using the saved model"""
    # Extract components
    model = model_package['best_model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # Ensure input data has the same features
    input_df = pd.DataFrame([input_data])
    
    # Align columns with training data
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default value
    
    # Reorder columns to match training
    input_df = input_df[feature_names]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_log = model.predict(input_scaled)[0]
    
    # Convert back from log scale
    prediction = np.expm1(prediction_log)
    
    return prediction

# Example usage
if __name__ == "__main__":
    # Load the model
    model_package = load_house_price_model()
    
    if model_package:
        # Example input (you would get this from your Streamlit app)
        example_input = {
            'GrLivArea': 1500,
            'TotalBsmtSF': 800,
            'OverallQual': 7,
            'YearBuilt': 2000,
            # ... include all other features with default values
        }
        
        # Fill missing features with median/mean values from your training
        for feature in model_package['feature_names']:
            if feature not in example_input:
                example_input[feature] = 0  # Use appropriate default
        
        # Make prediction
        price = predict_house_price(model_package, example_input)
        print(f"🏠 Predicted House Price: ${price:,.2f}")