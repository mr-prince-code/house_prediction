"""
House Price Model Training Script
Trains the model and saves it as PKL files for use in web app
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HOUSE PRICE MODEL TRAINING & SAVING")
print("=" * 60)

# ============================================
# STEP 1: Load and Prepare Data
# ============================================
print("\n📂 Loading data...")

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")

# Save target and IDs
y_train = train_df['SalePrice']
train_df = train_df.drop('SalePrice', axis=1)
test_ids = test_df['Id']

# Combine for preprocessing
all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ============================================
# STEP 2: Handle Missing Values
# ============================================
print("\n🧹 Cleaning data...")

# Categorical: NaN means "None"
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in none_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

# Numerical: NaN means 0
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
             'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# LotFrontage: Fill with median by neighborhood
if 'LotFrontage' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

# Fill remaining with mode/median
for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:
            all_data[col] = all_data[col].fillna(all_data[col].median())

print(f"✓ Missing values handled: {all_data.isnull().sum().sum()} remaining")

# ============================================
# STEP 3: Feature Engineering
# ============================================
print("\n🔧 Engineering features...")

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath'] + 
                          all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'])
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

print("✓ New features created")

# ============================================
# STEP 4: Encode Categorical Variables
# ============================================
print("\n🔤 Encoding categorical variables...")

# Drop Id
all_data = all_data.drop('Id', axis=1)

# Save column names before encoding
numerical_features = all_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = all_data.select_dtypes(include=['object']).columns.tolist()

# One-hot encode
all_data = pd.get_dummies(all_data, columns=categorical_features, drop_first=True)

print(f"✓ Encoded. Total features: {all_data.shape[1]}")

# ============================================
# STEP 5: Split Back to Train and Test
# ============================================
train_processed = all_data[:len(train_df)]
test_processed = all_data[len(train_df):]

# Remove outliers
outliers = train_processed[train_processed['GrLivArea'] > 4000].index
if len(outliers) > 0:
    train_processed = train_processed.drop(outliers)
    y_train = y_train.drop(outliers)
    print(f"✓ Removed {len(outliers)} outliers")

X_train = train_processed
y_train_log = np.log1p(y_train)  # Log transform

# ============================================
# STEP 6: Scale Features
# ============================================
print("\n⚖️ Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_processed)

print("✓ Scaling complete")

# ============================================
# STEP 7: Train Multiple Models
# ============================================
print("\n🤖 Training models...")

models = {
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.001, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_log, test_size=0.2, random_state=42)

results = {}
for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_tr, y_tr)
    
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)
    
    results[name] = {'RMSE': rmse, 'R2': r2, 'model': model}
    print(f"    RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ============================================
# STEP 8: Select Best Model
# ============================================
print("\n🏆 Selecting best model...")

best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['model']

print(f"✓ Best model: {best_model_name}")
print(f"  RMSE: {results[best_model_name]['RMSE']:.4f}")
print(f"  R²: {results[best_model_name]['R2']:.4f}")

# Retrain on full training data
print(f"\n🔄 Retraining {best_model_name} on full training data...")
best_model.fit(X_train_scaled, y_train_log)

# ============================================
# STEP 9: SAVE MODEL AND SCALER TO PKL
# ============================================
print("\n💾 Saving model files...")

# Save the trained model
with open('house_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: house_model.pkl")

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# Save feature names (important for prediction)
feature_names = X_train.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Saved: feature_names.pkl")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'rmse': results[best_model_name]['RMSE'],
    'r2': results[best_model_name]['R2'],
    'n_features': X_train.shape[1],
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✓ Saved: model_metadata.pkl")

# ============================================
# STEP 10: Make Predictions
# ============================================
print("\n🔮 Making predictions on test data...")

test_predictions = best_model.predict(X_test_scaled)
test_predictions = np.expm1(test_predictions)  # Reverse log transform

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("✓ Saved: submission.csv")

# ============================================
# STEP 11: Test Loading the Model
# ============================================
print("\n🧪 Testing model loading...")

try:
    # Load model
    with open('house_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    
    # Test prediction
    test_pred = loaded_model.predict(loaded_scaler.transform(X_train[:1]))
    print("✓ Model loads successfully!")
    print(f"  Test prediction: ${np.expm1(test_pred[0]):,.0f}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\n📦 Files saved:")
print("  1. house_model.pkl       - Trained model")
print("  2. scaler.pkl            - Feature scaler")
print("  3. feature_names.pkl     - Feature column names")
print("  4. model_metadata.pkl    - Model information")
print("  5. submission.csv        - Test predictions")

print(f"\n📊 Model Performance:")
print(f"  Algorithm: {best_model_name}")
print(f"  RMSE: {results[best_model_name]['RMSE']:.4f}")
print(f"  R² Score: {results[best_model_name]['R2']:.4f}")
print(f"  Features: {X_train.shape[1]}")

print("\n🚀 Ready to use in web app!")
print("   Load with: pickle.load(open('house_model.pkl', 'rb'))")