from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Function to calculate model metrics
def calculate_metrics(df, y_pred_final):
    y_true = df['vomitoxin_ppb'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_final))
    r2 = r2_score(y_true, y_pred_final)
    
    metrics = {
        'RMSE': rmse,
        'R²': r2
    }
    
    return metrics
