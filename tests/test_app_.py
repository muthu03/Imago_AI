import pytest
import io
import pandas as pd
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_train_endpoint(client):
    df = pd.read_csv("D:\Imago_AI\data\MLE-Assignment.csv")  # Replace with your actual CSV file
    csv_data = df.to_csv(index=False)
    
    response = client.post('/train', data={'file': (io.BytesIO(csv_data.encode()), 'data.csv')})
    
    assert response.status_code == 200
    assert "Model training completed" in response.json["message"]

def test_predict_endpoint(client):
    df = pd.read_csv("D:\Imago_AI\data\MLE-Assignment.csv")  # Read CSV file
    first_row = df.head(1)  # Take only the first row
    csv_data = first_row.to_csv(index=False)  # Convert to CSV format
    
    response = client.post('/predict', data={'file': (io.BytesIO(csv_data.encode()), 'data.csv')})
    
    assert response.status_code == 200
    assert "predictions" in response.json
    assert "metrics" in response.json
