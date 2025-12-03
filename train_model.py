import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data
try:
    df = pd.read_csv("Data/final_data.csv")
    print("Data loaded successfully with shape:", df.shape)
    
    # Check if AQI and features are in the dataframe
    print("Columns:", df.columns.tolist())
    
    # Identify the feature columns and target column
    feature_cols = [col for col in df.columns if col not in ['AQI']]
    
    if 'AQI' in df.columns:
        # Basic preprocessing
        X = df[feature_cols]
        y = df['AQI']
        
        # Train a simple RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save the model
        with open('./new_model.pkl', 'wb') as file:
            pickle.dump({"model": model}, file)
        
        print("Model trained and saved successfully!")
        print("Features used:", feature_cols)
    else:
        print("Error: 'AQI' column not found in the dataset.")
        
except Exception as e:
    print(f"Error: {e}") 