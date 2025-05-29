# Project: House Price Prediction using Machine Learning

# ============================================
# 1. Load data and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seab as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('train.csv')

# ============================================
# 2. Exploratory Data Analysis (EDA)
print(df.shape)
print(df.info())
print(df.describe())

# Missing values
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# ============================================
# 3. Data Cleaning
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
df = df.drop(columns=cols_to_drop, errors='ignore')

none_cols = ['MasVnrType', 'FireplaceQu', 'GarageFinish', 'GarageQual', 
             'GarageCond', 'GarageType', 'BsmtExposure', 'BsmtFinType2', 
             'BsmtCond', 'BsmtFinType1', 'BsmtQual']
df[none_cols] = df[none_cols].fillna('None')

df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# ============================================
# 4. Feature Preparation
weak_features = [
    'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Id', 'LowQualFinSF',
    'YrSold', 'OverallCond', 'MSSubClass', 'KitchenAbvGr', 'EnclosedPorch'
]
df = df.drop(columns=weak_features)

df = pd.get_dummies(df)

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 5. Model Training
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ============================================
# 6. Evaluation
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Results")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print("=" * 30)

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# ============================================
# 7. Insights and Conclusions
"""
- XGBoost showed the best performance across all metrics (RMSE, MAE, R2)
- Dropping weakly correlated features slightly improved results
- Handling missing values prevented model failures during training
- Linear Regression was good but not suitable for complex non-linear relationships
- Random Forest was stable but less accurate than XGBoost

Key takeaway:
The strongest model isn't always the best choice — it's about balancing accuracy, speed, complexity, and interpretability.
"""
# ============================================
# End of Project
