import pandas as pd
import joblib
import os
from modules.model_creation import train_models, load_models

def test_train_models():
  
    df = pd.read_csv('D:\Imago_AI\data\MLE-Assignment.csv')

    # Train models
    message = train_models(df)

    # Check if models are saved
    assert os.path.exists('model_classification.pkl')
    assert os.path.exists('model_regression.pkl')
    assert os.path.exists('scaler.pkl')
    assert message == "Model training completed successfully!"

    # Load models
    clf, reg_xgb, scaler = load_models()
    assert clf is not None
    assert reg_xgb is not None
    assert scaler is not None
