import pandas as pd
import numpy as np

# Function to preprocess the data
def preprocess_data(df):
    # Keep only numeric columns (selecting appropriate columns)
    df = df.select_dtypes(include=[np.number])
    df['log_vomitoxin'] = np.log1p(df['vomitoxin_ppb'])
    df['is_contaminated'] = (df['vomitoxin_ppb'] > 0).astype(int)
    return df
