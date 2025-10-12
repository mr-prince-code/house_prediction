"""
House Price Model Training with Hyperparameter Tuning
Uses GridSearchCV and RandomizedSearchCV for optimization
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HOUSE PRICE MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("=" * 70)

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
# STEP 2: Data Preprocessing (Same as before)
# ============================================
print("\n🧹 Cleaning data...")

# Handle missing values
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in none_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
             'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

if 'LotFrontage' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:
            all_data[col] = all_data[col].fillna(all_data[col].median())

# Feature Engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath'] + 
                          all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'])
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

# Encoding
all_data = all_data.drop('Id', axis=1)
categorical_features = all_data.select_dtypes(include=['object']).columns.tolist()
all_data = pd.get_dummies(all_data, columns=categorical_features, drop_first=True)

print(f"✓ Total features after preprocessing: {all_data.shape[1]}")

# Split back
train_processed = all_data[:len(train_df)]
test_processed = all_data[len(train_df):]

# Remove outliers
outliers = train_processed[train_processed['GrLivArea'] > 4000].index
if len(outliers) > 0:
    train_processed = train_processed.drop(outliers)
    y_train = y_train.drop(outliers)
    print(f"✓ Removed {len(outliers)} outliers")

X_train = train_processed
y_train_log = np.log1p(y_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_processed)

# Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_log, 
                                            test_size=0.2, random_state=42)

print(f"Training set: {X_tr.shape}")
print(f"Validation set: {X_val.shape}")

# ============================================
# STEP 3: HYPERPARAMETER TUNING
# ============================================
print("\n" + "=" * 70)
print("🔧 HYPERPARAMETER TUNING")
print("=" * 70)

# Define scoring metric (negative MSE for minimization)
rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))

tuning_results = {}

# ============================================
# 3.1 LASSO - GridSearchCV
# ============================================
print("\n🔍 Tuning Lasso Regression with GridSearchCV...")

lasso_param_grid = {
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'max_iter': [10000, 20000],
    'tol': [0.0001, 0.001]
}

lasso_grid = GridSearchCV(
    Lasso(random_state=42),
    param_grid=lasso_param_grid,
    cv=5,
    scoring=rmse_scorer,
    n_jobs=-1,
    verbose=1
)

lasso_grid.fit(X_tr, y_tr)

print(f"✓ Best parameters: {lasso_grid.best_params_}")
print(f"✓ Best CV RMSE: {-lasso_grid.best_score_:.4f}")

# Evaluate on validation set
lasso_pred = lasso_grid.predict(X_val)
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_pred))
lasso_r2 = r2_score(y_val, lasso_pred)

tuning_results['Lasso'] = {
    'model': lasso_grid.best_estimator_,
    'best_params': lasso_grid.best_params_,
    'cv_rmse': -lasso_grid.best_score_,
    'val_rmse': lasso_rmse,
    'val_r2': lasso_r2,
    'method': 'GridSearchCV'
}

print(f"✓ Validation RMSE: {lasso_rmse:.4f}")
print(f"✓ Validation R²: {lasso_r2:.4f}")

# ============================================
# 3.2 RIDGE - GridSearchCV
# ============================================
print("\n🔍 Tuning Ridge Regression with GridSearchCV...")

ridge_param_grid = {
    'alpha': [0.1, 1, 5, 10, 20, 50, 100],
    'max_iter': [10000],
    'tol': [0.0001, 0.001]
}

ridge_grid = GridSearchCV(
    Ridge(random_state=42),
    param_grid=ridge_param_grid,
    cv=5,
    scoring=rmse_scorer,
    n_jobs=-1,
    verbose=1
)

ridge_grid.fit(X_tr, y_tr)

print(f"✓ Best parameters: {ridge_grid.best_params_}")
print(f"✓ Best CV RMSE: {-ridge_grid.best_score_:.4f}")

ridge_pred = ridge_grid.predict(X_val)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
ridge_r2 = r2_score(y_val, ridge_pred)

tuning_results['Ridge'] = {
    'model': ridge_grid.best_estimator_,
    'best_params': ridge_grid.best_params_,
    'cv_rmse': -ridge_grid.best_score_,
    'val_rmse': ridge_rmse,
    'val_r2': ridge_r2,
    'method': 'GridSearchCV'
}

print(f"✓ Validation RMSE: {ridge_rmse:.4f}")
print(f"✓ Validation R²: {ridge_r2:.4f}")

# ============================================
# 3.3 RANDOM FOREST - RandomizedSearchCV
# ============================================
print("\n🔍 Tuning Random Forest with RandomizedSearchCV...")

rf_param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=20,  # Try 20 random combinations
    cv=3,  # 3-fold CV (faster than 5)
    scoring=rmse_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_random.fit(X_tr, y_tr)

print(f"✓ Best parameters: {rf_random.best_params_}")
print(f"✓ Best CV RMSE: {-rf_random.best_score_:.4f}")

rf_pred = rf_random.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
rf_r2 = r2_score(y_val, rf_pred)

tuning_results['RandomForest'] = {
    'model': rf_random.best_estimator_,
    'best_params': rf_random.best_params_,
    'cv_rmse': -rf_random.best_score_,
    'val_rmse': rf_rmse,
    'val_r2': rf_r2,
    'method': 'RandomizedSearchCV'
}

print(f"✓ Validation RMSE: {rf_rmse:.4f}")
print(f"✓ Validation R²: {rf_r2:.4f}")

# ============================================
# 3.4 GRADIENT BOOSTING - RandomizedSearchCV
# ============================================
print("\n🔍 Tuning Gradient Boosting with RandomizedSearchCV...")

gb_param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2']
}

gb_random = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gb_param_dist,
    n_iter=20,
    cv=3,
    scoring=rmse_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

gb_random.fit(X_tr, y_tr)

print(f"✓ Best parameters: {gb_random.best_params_}")
print(f"✓ Best CV RMSE: {-gb_random.best_score_:.4f}")

gb_pred = gb_random.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
gb_r2 = r2_score(y_val, gb_pred)

tuning_results['GradientBoosting'] = {
    'model': gb_random.best_estimator_,
    'best_params': gb_random.best_params_,
    'cv_rmse': -gb_random.best_score_,
    'val_rmse': gb_rmse,
    'val_r2': gb_r2,
    'method': 'RandomizedSearchCV'
}

print(f"✓ Validation RMSE: {gb_rmse:.4f}")
print(f"✓ Validation R²: {gb_r2:.4f}")

# ============================================
# STEP 4: MODEL COMPARISON
# ============================================
print("\n" + "=" * 70)
print("📊 TUNED MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame({
    'Model': list(tuning_results.keys()),
    'Tuning Method': [tuning_results[m]['method'] for m in tuning_results.keys()],
    'CV RMSE': [tuning_results[m]['cv_rmse'] for m in tuning_results.keys()],
    'Val RMSE': [tuning_results[m]['val_rmse'] for m in tuning_results.keys()],
    'Val R²': [tuning_results[m]['val_r2'] for m in tuning_results.keys()]
})

results_df = results_df.sort_values('Val RMSE')
print("\n" + results_df.to_string(index=False))

# ============================================
# STEP 5: SELECT BEST MODEL
# ============================================
best_model_name = results_df.iloc[0]['Model']
best_model_info = tuning_results[best_model_name]
best_model = best_model_info['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Tuning Method: {best_model_info['method']}")
print(f"   Best Parameters: {best_model_info['best_params']}")
print(f"   Cross-Validation RMSE: {best_model_info['cv_rmse']:.4f}")
print(f"   Validation RMSE: {best_model_info['val_rmse']:.4f}")
print(f"   Validation R²: {best_model_info['val_r2']:.4f}")

# Retrain on full training data
print(f"\n🔄 Retraining {best_model_name} on full training data...")
best_model.fit(X_train_scaled, y_train_log)

# ============================================
# STEP 6: SAVE EVERYTHING
# ============================================
print("\n💾 Saving model files...")

# Save model
with open('house_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: house_model_tuned.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)
print("✓ Saved: feature_names.pkl")

# Save metadata with tuning info
metadata = {
    'model_name': best_model_name,
    'tuning_method': best_model_info['method'],
    'best_params': best_model_info['best_params'],
    'cv_rmse': best_model_info['cv_rmse'],
    'val_rmse': best_model_info['val_rmse'],
    'val_r2': best_model_info['val_r2'],
    'n_features': X_train.shape[1],
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('model_metadata_tuned.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✓ Saved: model_metadata_tuned.pkl")

# Save all tuning results
with open('tuning_results.pkl', 'wb') as f:
    pickle.dump(tuning_results, f)
print("✓ Saved: tuning_results.pkl")

# ============================================
# STEP 7: MAKE PREDICTIONS
# ============================================
print("\n🔮 Making predictions on test data...")

test_predictions = best_model.predict(X_test_scaled)
test_predictions = np.expm1(test_predictions)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})
submission.to_csv('submission_tuned.csv', index=False)
print("✓ Saved: submission_tuned.csv")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ HYPERPARAMETER TUNING COMPLETE!")
print("=" * 70)

print("\n📊 Summary:")
print(f"   Best Model: {best_model_name}")
print(f"   Tuning Technique: {best_model_info['method']}")
print(f"   Validation R²: {best_model_info['val_r2']:.4f} ({best_model_info['val_r2']*100:.2f}%)")
print(f"   Validation RMSE: {best_model_info['val_rmse']:.4f}")

print("\n🔧 Tuning Methods Used:")
print("   ✓ GridSearchCV - Exhaustive search (Lasso, Ridge)")
print("   ✓ RandomizedSearchCV - Random search (Random Forest, Gradient Boosting)")
print("   ✓ 5-Fold Cross-Validation for linear models")
print("   ✓ 3-Fold Cross-Validation for ensemble models (faster)")

print("\n📦 Files Created:")
print("   ✓ house_model_tuned.pkl")
print("   ✓ scaler.pkl")
print("   ✓ feature_names.pkl")
print("   ✓ model_metadata_tuned.pkl")
print("   ✓ tuning_results.pkl")
print("   ✓ submission_tuned.csv")

print("\n🎯 Best Hyperparameters Found:")
for param, value in best_model_info['best_params'].items():
    print(f"   • {param}: {value}")

print("\n✨ Model is now optimized and ready for deployment!")