import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data, filter_data
from utils import standardize, RMSE
from factor_model import DynamicFactorModel
from datetime import datetime

# Load and filter data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
df_data = load_data(file_path)

# Apply Christiano-Fitzgerald filter
filtered_df = filter_data(df_data)

# Inspect the filtered data
print(filtered_df.head())
print(filtered_df.columns)
print(filtered_df.shape)

# Save variable names
variable_names = filtered_df.index.tolist()

# Define number of factors
num_factors = 9

# Initialize the dynamic factor model
model = DynamicFactorModel(filtered_df, num_factors)

# Define validation date and split the data
DATE_VALIDATE = pd.Period('2020-01', freq='M')
if DATE_VALIDATE in filtered_df.columns:
    date_index = filtered_df.columns.get_loc(DATE_VALIDATE)
else:
    raise ValueError(f"Date {DATE_VALIDATE} not found in the columns of the dataframe")

# Prepare training data until 2019-12, Validation from 2020-01 to 2023-11 (47 months)
Y_train_PCA = filtered_df.iloc[:, :date_index]
Y_validate = filtered_df.iloc[:, date_index:date_index + 47]

# Standardize the datasets
Y_train_std = standardize(Y_train_PCA.values.T).T
Y_validate_std = standardize(Y_validate.values.T).T

# Fit the Dynamic Factor Model and apply PCA
model.std_data = Y_train_std.T
model.apply_pca()  # Apply the simpler PCA method

# Print shape of factors to ensure it matches expectations
print("Shape of PCA factors:", model.factors.shape)

# Estimate the Yule-Walker equations
model.yw_estimation()

# Prepare training and validation data for ElasticNet
train_split_index = int(model.factors.shape[1] * 0.8)

data_train = Y_train_std[:, :train_split_index].T
fac_train = model.factors[:, :train_split_index].T

data_validate = Y_validate_std.T
fac_validate = model.factors[:, train_split_index:train_split_index + 47].T  # Correctie toegepast

# Print shapes to debug potential dimension mismatches
print("Shape of data_train:", data_train.shape)
print("Shape of fac_train:", fac_train.shape)
print("Shape of data_validate:", data_validate.shape)
print("Shape of fac_validate:", fac_validate.shape)

B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

# Validate model
y_hat_validate = model.enet_predict(fac_validate)

# Print shapes before RMSE calculation
print("Shape of y_hat_validate:", y_hat_validate.shape)
print("Shape of data_validate:", data_validate.shape)

# Calculate RMSE for validation data
try:
    rmse_value = RMSE(data_validate, y_hat_validate)
    # Ensure variable names match RMSE values length
    valid_variable_names = variable_names[:len(rmse_value)]
    # Create a DataFrame with RMSE values and variable names
    rmse_table = pd.DataFrame({'Variable': valid_variable_names, 'RMSE': rmse_value})
    print(rmse_table)
except ValueError as e:
    print(f"RMSE calculation error: {e}")
    print(f"Shape mismatch details - data_validate: {data_validate.shape}, y_hat_validate: {y_hat_validate.shape}")

# Print additional results
print(f"R2 in-sample: {r2_insample}")
print(f"ElasticNet intercept: {intercept}")

# Confirm the script has finished
print("Script execution completed.")

rmse_table.to_excel('rmse_static.xlsx', index=False)