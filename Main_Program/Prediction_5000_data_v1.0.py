import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to load data from SQLite database
def load_data(db_path, table_name):
    """
    Connects to the SQLite database and loads the data from the specified table.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql(query, conn)
        conn.close()
        return data
    except sqlite3.OperationalError as e:
        print(f"Error: {e}. Please ensure the table '{table_name}' exists in the database.")
        return None

# Function to preprocess the data
def preprocess_data(data):
    """
    Preprocesses the data by encoding categorical variables, handling missing columns, and scaling numerical features.
    """
    # Check if 'Date' column exists
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data.drop(columns=['Date'], inplace=True)

    # Encode categorical variables
    categorical_columns = [
        'Factory', 'Location', 'Machine Type', 'Operator Training Level', 'Raw Material Quality',
        'Maintenance History', 'Defect Root Cause', 'Energy Efficiency Rating', 'Emission Limit Compliance',
        'Shift', 'Product Category', 'Supplier'
    ]
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    return data

# Function to evaluate and choose the best regression method
def choose_best_regression_method(X, y):
    """
    Evaluates and chooses the best regression method based on cross-validation scores.
    """
    methods = {
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1)
    }

    best_method = None
    best_score = float('-inf')
    for name, model in methods.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        print(f"Method: {name}, Cross-Validation R2 Score: {mean_score:.2f}")
        if mean_score > best_score:
            best_score = mean_score
            best_method = model

    print(f"\nBest Method Selected: {best_method.__class__.__name__} with Confidence: {best_score:.2f}\n")
    return best_method, best_score

# Function to train the selected regression model
def train_model(X, y, model):
    """
    Trains the selected regression model and returns the trained model.
    """
    model.fit(X, y)
    return model

# Function to identify significant parameters affecting predictions
def get_significant_parameters(model, feature_columns):
    """
    Identifies and returns the most significant parameters affecting the predictions using feature importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        significant_params = pd.Series(importances, index=feature_columns)
        return significant_params.sort_values(ascending=False)
    else:
        return pd.Series(dtype=float)

# Function to predict for the next 6 months and save to CSV with different files
def predict_next_six_months(model, data, feature_columns, target):
    """
    Predicts target variable for all factories and locations for the next 6 months (Jan 2025 - June 2025)
    and saves the results to a CSV file.
    """
    # Get unique factories and locations
    factories = data['Factory'].unique()
    locations = data['Location'].unique()

    # Create a dataframe for the next 6 months predictions
    future_data = pd.DataFrame()
    for factory in factories:
        for location in locations:
            temp_data = pd.DataFrame({
                'Year': [2025] * 6,
                'Month': [1, 2, 3, 4, 5, 6],
                'Factory': [factory] * 6,
                'Location': [location] * 6
            })
            for col in feature_columns:
                if col not in temp_data.columns:
                    temp_data[col] = data[col].mean()

            future_data = pd.concat([future_data, temp_data], ignore_index=True)

    predictions = model.predict(future_data[feature_columns])
    future_data[f'Predicted {target}'] = predictions

    output_file = f'predictions_{target.replace(" ", "_").lower()}.csv'
    future_data.to_csv(output_file, index=False)
    print(f"Predictions for {target} have been saved to {output_file}.")

    # Plotting predictions for each factory and location
    plt.figure(figsize=(12, 8))
    for factory in factories:
        factory_data = future_data[future_data['Factory'] == factory]
        for location in locations:
            location_data = factory_data[factory_data['Location'] == location]
            plt.plot(
                location_data['Month'],
                location_data[f'Predicted {target}'],
                marker='o',
                label=f'Factory {factory}, Location {location}'
            )

    plt.title(f"{target} Predictions (Jan 2025 - June 2025)")
    plt.xlabel("Month")
    plt.ylabel(target)
    plt.grid()
    plt.legend()
    plt.show()

    return future_data

# Function to calculate relationships between independent and dependent variables
def calculate_relationships(data, independent_columns, dependent_column):
    """
    Calculates and lists the relationships (correlations) between independent and dependent variables.
    """
    correlations = {}
    for col in independent_columns:
        correlations[col] = data[col].corr(data[dependent_column])

    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nRelationships between Independent and Dependent Variables for {dependent_column}:")
    for col, corr in sorted_correlations:
        print(f"{col}: Correlation = {corr:.2f}")

    return correlations

# Function to interactively predict and analyze parameters for multiple targets
def interactive_prediction():
    """
    Allows the user to interactively predict targets and analyze parameters.
    """
    # Database and table details
    db_path = r'D:\AIML-Richard\Data_Base\Factory_Data.db'
    table_name = "Sample_Data_5000_v1"

    # Load and preprocess the data
    data = load_data(db_path, table_name)
    if data is None or data.empty:
        print("Error: Data could not be loaded or is empty.")
        return

    # Targets to predict
    targets = ['Production Volume (units)', 'Revenue ($)', 'Foam Density']

    for target in targets:
        if target not in data.columns:
            print(f"Error: '{target}' column is missing from the data.")
            continue

        data = preprocess_data(data)

        # Identify independent and dependent variables
        independent_variables = data.drop(columns=[target], errors='ignore').columns

        # Define features and target
        X = data[independent_variables]
        y = data[target]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Automatically choose the best regression method
        model, best_score = choose_best_regression_method(X_scaled, y)
        model = train_model(X_scaled, y, model)

        # Identify significant parameters
        significant_params = get_significant_parameters(model, independent_variables)
        if not significant_params.empty:
            print(f"\nMost Significant Parameters Affecting {target}:")
            print(significant_params.head(3))
        else:
            print(f"No significant parameters identified for {target}.")

        # Predict for the next 6 months for all factories and locations
        predict_next_six_months(model, data, independent_variables, target)

        # Calculate relationships
        calculate_relationships(data, independent_variables, target)

# Execute the interactive prediction function
if __name__ == "__main__":
    interactive_prediction()