import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import os
from modules.data_pre_processing import preprocess_data
import logging
logger = logging.getLogger(__name__)

# Function to train models
def train_models(df):

    df = preprocess_data(df)
    logger.info('data pre-procesing done')
    
    # Prepare the feature matrix and target variable
    X = df.drop(columns=['vomitoxin_ppb', 'log_vomitoxin', 'is_contaminated'])
    y_class = df['is_contaminated']
    
    # Split data into train and test sets for classification
    logger.info("Data Splitted")
    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    # Train the classification model
    clf = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, reg_lambda=10, reg_alpha=5, random_state=42)
    clf.fit(X_train, y_class_train)
    
    # Filter only contaminated samples for regression
    df_contaminated = df[df['is_contaminated'] == 1]
    X_reg = df_contaminated.drop(columns=['vomitoxin_ppb', 'log_vomitoxin', 'is_contaminated'])
    y_reg = df_contaminated['log_vomitoxin']
    
    # Standard scaling of regression features
    scaler = StandardScaler()
    X_reg_scaled = scaler.fit_transform(X_reg)
    
    # Split data into train and test sets for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
    
    # Train the regression model
    reg_xgb = XGBRegressor(colsample_bytree=0.7, learning_rate=0.01, max_depth=6, n_estimators=500, subsample=0.7, random_state=42)
    reg_xgb.fit(X_train_reg, y_train_reg)
    logger.info("Model Trained")

    
    # Save models to disk
    joblib.dump(clf, 'model_classification.pkl')
    joblib.dump(reg_xgb, 'model_regression.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    logger.info("Model Saved")
    
    return "Model training completed successfully!"

# Function to load models
def load_models():
    if os.path.exists('model_classification.pkl') and os.path.exists('model_regression.pkl') and os.path.exists('scaler.pkl'):
        return joblib.load('model_classification.pkl'), joblib.load('model_regression.pkl'), joblib.load('scaler.pkl')
    return None, None, None
