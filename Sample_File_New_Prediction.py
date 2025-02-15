import pandas as pd
import pickle
import numpy as np

def load_models(pkl_file_path):
    """Load trained models from the pickle file."""
    with open(pkl_file_path, 'rb') as file:
        models = pickle.load(file)
    return models

def prepare_input_data(features, scaler, input_values):
    """Prepare input data by scaling and structuring it properly."""
    input_df = pd.DataFrame([input_values], columns=features)
    input_scaled = scaler.transform(input_df)
    return input_scaled

def make_prediction(models, location, target_variable, input_values):
    """Run prediction for a given location and target variable."""
    model_info = models.get((location, target_variable))
    if model_info is None:
        raise ValueError(f"No trained model found for location '{location}' and target '{target_variable}'")
    
    model = model_info['model']
    scaler = model_info['scaler']
    features = model_info['features']
    
    input_scaled = prepare_input_data(features, scaler, input_values)
    prediction = model.predict(input_scaled)
    return prediction[0]

if __name__ == "__main__":
    # Load trained models
    model_file_path = "all_models.pkl"
    models = load_models(model_file_path)
    
    # Example input values for prediction
    location = 2  # Example location encoded value
    target_variable = "Production Volume (units)"
    input_values = {
        "Machine Utilization (%)": 80.0,
        "Batch Quality (Pass %)": 95.0,
        "Breakdowns (count)": 3,
        "Machine Age (years)": 5.0
    }
    
    # Ensure input values match expected features
    model_info = models.get((location, target_variable))
    if model_info:
        features = model_info['features']
        input_values = [input_values.get(feat, np.nan) for feat in features]
        
        # Run prediction
        prediction_result = make_prediction(models, location, target_variable, input_values)
        print(f"Predicted {target_variable}: {prediction_result}")
    else:
        print(f"No model available for location: {location} and target variable: {target_variable}")
