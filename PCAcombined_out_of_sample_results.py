import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_combined_data, filter_data
from utils import standardize, RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model import DynamicFactorModel

# Zorg ervoor dat de directory bestaat waar we de plots gaan opslaan
plot_dir = "plots_PCAcombined"
os.makedirs(plot_dir, exist_ok=True)

# Load and filter combined data
static_file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
forward_file_path = 'C:/Thesis/03. Data/Final version data/Forward.xlsx'
combined_df = load_combined_data(static_file_path, forward_file_path)

# Apply Christiano-Fitzgerald filter
filtered_combined_df = filter_data(combined_df)

# Save variable names
variable_names = filtered_combined_df.index.tolist()

# Define training and validation periods and split the data
DATE_TRAIN_END = pd.Period('2019-12', freq='M')
DATE_VALIDATE_START = pd.Period('2020-01', freq='M')
DATE_VALIDATE_END = pd.Period('2023-11', freq='M')

# Split the data into training and validation sets
Y_train = filtered_combined_df.loc[:, :DATE_TRAIN_END]  # Data until 2019-12
Y_validate = filtered_combined_df.loc[:, DATE_VALIDATE_START:DATE_VALIDATE_END]  # Data from 2020-01 to 2023-11

# Standardize the datasets
Y_train_std = standardize(Y_train.values.T).T
Y_validate_std = standardize(Y_validate.values.T).T

# Load the predicted variables
predicted_df = pd.read_excel("C:/Thesis/04. Models/PCAcombined_predicted_variables_matrix_12.xlsx")

# Calculate RMSE and R-squared
rmse_values = RMSE(Y_validate_std, predicted_df)
r2_values = calculate_r2(Y_validate_std, predicted_df)

# Calculate log-likelihood
log_like = log_likelihood(Y_validate_std.flatten(), predicted_df.values.flatten())
print(f"Log-Likelihood: {log_like}")

# Number of parameters in your model (e.g., number of factors + intercept)
num_params = 13  # Adjust this based on your model

# Calculate AIC and BIC
aic, bic = calculate_aic_bic(predicted_df.values.flatten(), Y_validate_std.flatten(), num_params)
print(f"AIC: {aic}")
print(f"BIC: {bic}")

# Calculate adjusted R²
r2_mean = r2_values.mean()  # Use the mean R² across all columns
n = len(Y_validate_std.flatten())  # Total number of observations
p = num_params  # Number of parameters (factors)
adj_r2 = adjusted_r2(r2_mean, n, p)
print(f"Adjusted R²: {adj_r2}")

# Combineer de resultaten
results_df = pd.DataFrame({
    'RMSE': rmse_values,
    'R_squared': r2_values,
    'Log_Likelihood': [log_like] * len(rmse_values),
    'AIC': [aic] * len(rmse_values),
    'BIC': [bic] * len(rmse_values),
    'Adjusted_R_squared': [adj_r2] * len(rmse_values)
})

# Print de resultaten
print(results_df)

# Opslaan van resultaten in een Excel-bestand
results_df.to_excel('PCAcombined_out_of_sample_metrics_factor_12.xlsx', index=False)
