import pandas as pd
import numpy as np
from modules.model_metrics import calculate_metrics

def test_calculate_metrics():
    # True values
    df =pd.read_csv('D:\Imago_AI\data\MLE-Assignment.csv')

    # Predicted values
    y_pred_final = np.array([12, 18, 29, 39, 51])

    # Compute metrics
    metrics = calculate_metrics(df, y_pred_final)

    assert "RMSE" in metrics
    assert "R²" in metrics
    assert metrics["RMSE"] > 0
