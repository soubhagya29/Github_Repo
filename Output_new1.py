import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Load and Preprocess Data
file_path = 'D:\Foam_Factory\synthetic_factory_data_all_outputs.csv'  # Update if needed

data = pd.read_csv(file_path)

# Convert Date column if exists
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data.drop(columns=['Date'], inplace=True)

# Encode categorical columns
categorical_columns = ['Factory', 'Location', 'Machine Type', 'Operator Training Level', 'Raw Material Quality',
                       'Maintenance History', 'Supplier', 'Shift', 'Batch']
label_encoders = {}
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# Handle missing values
imputer = SimpleImputer(strategy='mean')
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Feature Selection and Target Definitions
feature_target_pairs = {
    'Production Volume (units)': ['Machine Utilization (%)', 'Batch Quality (Pass %)', 'Breakdowns (count)',
                                  'Machine Age (years)'],
    'Machine Utilization (%)': ['Location', 'Machine Type', 'Batch', 'Shift', 'Operator Training Level', 'Supplier'],
    'Revenue ($)': ['Raw Material Quality', 'Supplier Delays (days)', 'Supplier', 'Market Demand Index'],
    'Profit Margin (%)': ['Cost of Downtime ($)', 'Breakdowns (count)', 'Safety Incidents (count)',
                          'Production Volume (units)']
}

# Store trained models and R2 scores
predictions_output = {}
r2_scores_log = []

# Train models per location
for location in data['Location'].unique():
    location_data = data[data['Location'] == location]

    for target, features in feature_target_pairs.items():
        if target in location_data.columns and all(f in location_data.columns for f in features):
            X = location_data[features]
            y = location_data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "LinearRegression": LinearRegression()
            }

            best_model_name = None
            best_model_score = float('-inf')
            best_model = None

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                r2_scores_log.append(f"{location} - {name} R2 Score for '{target}': {score:.4f}")
                print(f"{location} - {name} R2 Score for '{target}': {score:.4f}")
                if score > best_model_score:
                    best_model_score = score
                    best_model_name = name
                    best_model = model

            predictions_output[(location, target)] = {
                'model': best_model,
                'scaler': scaler,
                'features': features
            }

# Save all models in a single pkl file
with open('all_models.pkl', 'wb') as file:
    pickle.dump(predictions_output, file)

# Save R2 scores to a log file
with open('model_r2_scores.txt', 'w') as log_file:
    for entry in r2_scores_log:
        log_file.write(entry + "\n")

print("All models trained and saved successfully in a single pkl file: all_models.pkl")
