import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor

# Create a new empty model with the current scikit-learn version
new_model = RandomForestRegressor()

try:
    # Attempt to load the old model data (this will fail but we'll catch the exception)
    with open('./model.pkl', 'rb') as file:
        try:
            old_data = pickle.load(file)
            print("Successfully loaded old model")
        except Exception as e:
            print(f"Error loading old model: {e}")
            print("Will create a new model instead")
            old_data = {"model": new_model}
    
    # Save with the current scikit-learn version
    with open('./new_model.pkl', 'wb') as file:
        pickle.dump(old_data, file)
    
    print("Created new model file: new_model.pkl")
    
except Exception as e:
    print(f"Error: {e}") 