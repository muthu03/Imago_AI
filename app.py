from flask import Flask, request, jsonify
import pandas as pd
from modules.data_pre_processing import preprocess_data
from modules.model_creation import train_models, load_models
from modules.model_metrics import calculate_metrics
import joblib
import numpy as np
import logging

# Configure logging
log_file = "app.log"  # Log file location
logging.basicConfig(
    level=logging.DEBUG,  # You can change this to INFO, WARNING, ERROR, etc.
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()  # Log to console (optional)
    ]
)

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    logging.info("Model Training Started")
    df = pd.read_csv(file)
    message = train_models(df)
    return jsonify({"message": message})

@app.route('/predict', methods=['POST'])
def predict():
    clf, reg_xgb, scaler = load_models()
    if clf is None or reg_xgb is None or scaler is None:
        logging.error("Model not trained")
        return jsonify({"error": "Models are not trained yet."}), 400
    
    file = request.files['file']
    df = pd.read_csv(file)
    logging.info("Data Read Succesfully")
    df = preprocess_data(df)
    logging.info("Data Preprocessed")
    X = df.drop(columns=['vomitoxin_ppb', 'log_vomitoxin', 'is_contaminated'])
    y_class_pred = clf.predict(X)
    y_pred_final = np.zeros_like(y_class_pred, dtype=float)
    
    contaminated_indices = np.where(y_class_pred == 1)[0]
    if len(contaminated_indices) > 0:
        X_test_reg_selected = scaler.transform(X.iloc[contaminated_indices])
        y_reg_pred_log = reg_xgb.predict(X_test_reg_selected)
        y_pred_final[contaminated_indices] = np.expm1(y_reg_pred_log)
    
    metrics = calculate_metrics(df, y_pred_final)
    logging.info("metrics Calculated")
    return jsonify({"predictions": y_pred_final.tolist(), "metrics": metrics})

if __name__ == '__main__':
    app.run(debug=True)
