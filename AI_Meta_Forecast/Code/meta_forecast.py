"""
Meta-Forecasting Production Script: DELNET & DPSO
Author: Prashanth

This script performs rolling window meta-forecasting using two approaches:
1. DELNET (Dynamic ElasticNet)
2. DPSO (Dynamic Particle Swarm Optimization for weight optimization)

It calculates RMSE and MSE for each day window and saves results to Excel files in the 'Results/' folder.
Data is expected to be in the 'Data/' folder.

Data is assumed to be provided by the user.Please check and preprocess data
as required before running the script.
"""

# ---------------------------
# Required Libraries
# ---------------------------
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV  # Elasticnet import
from pyMetaheuristic.algorithm import particle_swarm_optimization # PSO import


# ---------------------------
# Configuration
# ---------------------------
np.random.seed(42)
DATA_FILE = "Data/YOUR_DATA_FILE.xlsx"  # Replace with your data file
DATE_COLUMN = "TS"                 # Name of the date column
TARGET_COLUMN = "Infeed [%]"       # Name of the target column
FORECAST_COLUMNS = [2, 3]          # Column indices of individual forecasts

WINDOW_SIZE = 15936                # Rolling window size of 165 days
STEP_SIZE = 96                     # Step size of 1 day for rolling window 
CV_FOLDS = 10                      # Cross-validation folds for ElasticNet
L1_RATIO = 0.5                     # ElasticNet l1_ratio


# ---------------------------
# Data Loading Function
# ---------------------------
def load_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_excel(file_path)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], dayfirst=True, format='mixed')
    return data

# ---------------------------
# Data Check Guidance
# ---------------------------
def check_data_guidance(data):
    """
    Guidance for users:
    - Check for missing values and decide how to handle them.
    - Remove duplicate rows if necessary.
    - Verify that the datetime column is correctly parsed.
    - Ensure numeric forecast columns and target column have valid values.
    - Adjust preprocessing based on your dataset specifics.
    """
    print("***********************************************************")
    print("WARNING: Please check your data before running the forecasts!")
    print(" - Look for missing values in target and forecast columns.")
    print(" - Remove duplicates if applicable.")
    print(" - Ensure datetime column is correctly parsed.")
    print(" - Validate numeric columns for NaN or invalid values.")
    print(" - Adjust preprocessing according to your data.")
    print("***********************************************************\n")
    
    print("Sample of your data:")
    print(data.head())
    print("\nBasic info:")
    print(data.info())
    print("\nBasic statistics:")
    print(data.describe())

    # Check for missing values
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        print(f"WARNING: Your dataset contains {missing_count} missing values. Please preprocess before running.")
    else:
        print("No missing values detected.")

# ---------------------------
# DELNET  Forecast
# ---------------------------
def delnet_forecast(data):
    """Perform rolling window meta-forecast using ElasticNetCV."""
    
    date_column = data[DATE_COLUMN]
    individual_forecasts = data.iloc[:, FORECAST_COLUMNS[0]:FORECAST_COLUMNS[1]+1].values
    actual_values = data[TARGET_COLUMN].values

    rmse_list = []
    predictions_list = []

    for i in range(0, len(date_column) - WINDOW_SIZE + 1, STEP_SIZE):
        window_dates = date_column[i:i + WINDOW_SIZE]
        window_forecasts = individual_forecasts[i:i + WINDOW_SIZE, :]
        window_actuals = actual_values[i:i + WINDOW_SIZE]

        train_size = WINDOW_SIZE - STEP_SIZE
        X_train, X_test = window_forecasts[:train_size], window_forecasts[train_size:]
        y_train, y_test = window_actuals[:train_size], window_actuals[train_size:]
        X_train_dates, X_test_dates = window_dates[:train_size], window_dates[train_size:]

        # Fit ElasticNetCV
        model = ElasticNetCV(l1_ratio=L1_RATIO, cv=CV_FOLDS)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = np.mean((y_test - predictions) ** 2)
        rmse = np.sqrt(mse)
        test_date = X_test_dates.iloc[0]

        # Append to list instead of concatenating each iteration
        rmse_list.append({'i': i, 'dates': test_date, 'mse': mse, 'rmse': rmse,'coefficients': model.coef_.tolist()})
        predictions_list.append(pd.DataFrame({
            'i': [i]*len(X_test_dates),
            'date': X_test_dates,
            'predicted_value': predictions,
            'actual': y_test
        }))

        print(f"ElasticNetCV - Window {i}: RMSE = {rmse:.4f}, Coefficients = {model.coef_}")

    rmse_df = pd.DataFrame(rmse_list)
    predictions_df = pd.concat(predictions_list, ignore_index=True)

    return rmse_df, predictions_df

# ---------------------------
# DPSO Forecast
# ---------------------------
def dpso_forecast(data):
    """Perform rolling window meta-forecast using DPSO weight optimization."""
    
    date_column = data[DATE_COLUMN]
    individual_forecasts = data.iloc[:, FORECAST_COLUMNS[0]:FORECAST_COLUMNS[1]+1].values
    actual_values = data[TARGET_COLUMN].values

    rmse_list = []
    predictions_list = []

    num_forecasts = individual_forecasts.shape[1]

    for i in range(0, len(date_column) - WINDOW_SIZE + 1, STEP_SIZE):
        window_dates = date_column[i:i + WINDOW_SIZE]
        window_forecasts = individual_forecasts[i:i + WINDOW_SIZE, :]
        window_actuals = actual_values[i:i + WINDOW_SIZE]

        train_size = WINDOW_SIZE - STEP_SIZE
        X_train, X_test = window_forecasts[:train_size], window_forecasts[train_size:]
        y_train, y_test = window_actuals[:train_size], window_actuals[train_size:]
        X_train_dates, X_test_dates = window_dates[:train_size], window_dates[train_size:]

        # Objective function for PSO
        def objective_function(weights):
            weights = np.clip(weights, 0, 1)
            weights /= np.sum(weights)
            meta_forecast = np.dot(weights, X_train.T)
            return np.mean((y_train - meta_forecast) ** 2)

        # PSO parameters
        pso_params = {
            'target_function': objective_function,
            'swarm_size': 30,
            'min_values': np.zeros(num_forecasts),
            'max_values': np.ones(num_forecasts),
            'iterations': 50,
            'w': 0.9,
            'c1': 2,
            'c2': 2,
            'verbose': False
        }

        optimized_weights = particle_swarm_optimization(**pso_params)[:-1]
        optimized_weights = np.clip(optimized_weights, 0, 1)
        optimized_weights /= np.sum(optimized_weights)

        predictions = np.dot(optimized_weights, X_test.T)

        mse = np.mean((y_test - predictions) ** 2)
        rmse = np.sqrt(mse)
        test_date = X_test_dates.iloc[0]

        # Append to list instead of concatenating each iteration
        rmse_list.append({'i': i, 'dates': test_date, 'mse': mse, 'rmse': rmse, 'weights': optimized_weights.tolist()})
        predictions_list.append(pd.DataFrame({
            'i': [i]*len(X_test_dates),
            'date': X_test_dates,
            'predicted_value': predictions,
            'actual': y_test
        }))

        print(f"DPSO - Window {i}: RMSE = {rmse:.4f}, Weights = {optimized_weights}")
    
    rmse_df = pd.DataFrame(rmse_list)
    predictions_df = pd.concat(predictions_list, ignore_index=True)

    return rmse_df, predictions_df

# ---------------------------
# Save Results
# ---------------------------
def save_results(rmse_df, predictions_df, output_path):
    """
    Save RMSE and predictions to Excel. file.
    If output_path is None, defaults to {output_prefix}_RMSE.xlsx in the current directory.
    """
    rmse_year = rmse_df.groupby([
        rmse_df['dates'].dt.year.rename('year'),
        rmse_df['dates'].dt.month_name().rename('month')
    ])[['mse', 'rmse']].mean().reset_index()



    with pd.ExcelWriter(output_path) as writer:
        rmse_df.to_excel(writer, sheet_name='rmse', index=False)
        rmse_year.to_excel(writer, sheet_name='rmse_year', index=False)
        predictions_df.to_excel(writer, sheet_name='predictions', index=False)


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":

    # Check if the data file exists
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    # Load and check data
    data = load_data(DATA_FILE)
    check_data_guidance(data)
    
    # Create results folder if missing
    os.makedirs("Results", exist_ok=True)
    
    # Choose methods (you can run one or both)
    methods = ["DELNET", "DPSO"]  # Options: "DELNET", "DPSO"
    
    for method in methods:
        print(f"\n================ Running {method.upper()} =================\n")
        
        if method.lower() == "delnet":
            rmse_df, predictions_df = delnet_forecast(data)
            output_file = os.path.join("Results", "DELNET_RMSE.xlsx")
        elif method.lower() == "dpso":
            rmse_df, predictions_df = dpso_forecast(data)
            output_file = os.path.join("Results", "DPSO_RMSE.xlsx")
        else:
            print(f"Unknown method: {method}. Skipping.")
            continue
        
        # Save results into 'results' folder
        save_results(rmse_df, predictions_df, output_file)
        
        print(f"{method.upper()} results saved to: {output_file}")
    
    print("\nAll selected methods completed successfully.")
