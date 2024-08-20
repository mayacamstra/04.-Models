import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_combined_data, filter_data
from utils import standardize, RMSE
from factor_model import DynamicFactorModel
from datetime import datetime

# Load and filter combined data
static_file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
forward_file_path = 'C:/Thesis/03. Data/Final version data/Forward.xlsx'
combined_df = load_combined_data(static_file_path, forward_file_path)

# Apply Christiano-Fitzgerald filter
filtered_combined_df = filter_data(combined_df)

# Separate static variables for RMSE calculation
static_df = combined_df.loc[combined_df.index[:66]]

# Inspect the filtered combined data
print(filtered_combined_df.head())
print(filtered_combined_df.columns)
print(filtered_combined_df.shape)

# Save variable names
variable_names = static_df.index.tolist()

# Define number of factors
num_factors = 9

# Initialize the model with PLS method
model = DynamicFactorModel(filtered_combined_df, num_factors, method='PLS')

# Define validation date and split the data
DATE_VALIDATE = pd.Period('2020-01', freq='M')
if DATE_VALIDATE in filtered_combined_df.columns:
    date_index = filtered_combined_df.columns.get_loc(DATE_VALIDATE)
else:
    raise ValueError(f"Date {DATE_VALIDATE} not found in the columns of the dataframe")

# Prepare training data until 2019-12, Validation from 2020-01 to 2023-11 (47 months)
Y_train_PLS = filtered_combined_df.iloc[:, :date_index]
Y_validate = filtered_combined_df.iloc[:, date_index:date_index + 47]

# Standardize the datasets
Y_train_std = standardize(Y_train_PLS.values.T).T
Y_validate_std = standardize(Y_validate.values.T).T

# Fit the Dynamic Factor Model and apply PLS
model.std_data = Y_train_std.T
model.apply_pls(Y_train_std.T, Y_train_std.T)  # Apply the PLS method

# Transpose the factors to match the expected shape
model.factors = model.factors.T  # Transpose to get (9, 300)

# Ensure that the PLS factors have the correct shape
print("Shape of PLS factors:", model.factors.shape)  # Expected shape: (9, 300)

# Prepare training and validation data for ElasticNet
train_split_index = int(model.factors.shape[1] * 0.8)

data_train = Y_train_std[:, :train_split_index].T
fac_train = model.factors[:, :train_split_index].T  # Now should have shape (240, 9)

data_validate = Y_validate_std.T
fac_validate = model.factors[:, train_split_index:train_split_index + 47].T  # Shape should be (47, 9)

# Print shapes to debug potential dimension mismatches
print("Shape of data_train:", data_train.shape)  # Expected shape: (240, 95)
print("Shape of fac_train:", fac_train.shape)  # Expected shape: (240, 9)
print("Shape of data_validate:", data_validate.shape)  # Expected shape: (47, 95)
print("Shape of fac_validate:", fac_validate.shape)  # Expected shape: (47, 9)

B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

# Validate model
y_hat_validate = model.enet_predict(fac_validate)

# Print shapes before RMSE calculation
print("Shape of y_hat_validate:", y_hat_validate.shape)
print("Shape of data_validate:", data_validate.shape)

# Calculate RMSE for validation data using only the original 66 variables
try:
    rmse_value = RMSE(data_validate[:, :66], y_hat_validate[:, :66])
    # Ensure variable names match RMSE values length
    valid_variable_names = variable_names[:len(rmse_value)]
    # Create a DataFrame with RMSE values and variable names
    rmse_table = pd.DataFrame({'Variable': valid_variable_names, 'RMSE': rmse_value})
    print(rmse_table)
except ValueError as e:
    print(f"RMSE calculation error: {e}")
    print(f"Shape mismatch details - data_validate: {data_validate.shape}, y_hat_validate: {y_hat_validate.shape}")

# Save RMSE table to a separate file for PLS results
if rmse_table is not None:
    rmse_table.to_excel('rmse_combined_pls.xlsx', index=False)

# Print additional results
print(f"R2 in-sample: {r2_insample}")
print(f"ElasticNet intercept: {intercept}")

# Confirm the script has finished
print("Script execution completed.")
