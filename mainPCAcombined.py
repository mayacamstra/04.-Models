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

# Initialize the model
model = DynamicFactorModel(filtered_combined_df, num_factors)

# Define validation date and split the data
DATE_VALIDATE = pd.Period('2020-01', freq='M')
if DATE_VALIDATE in filtered_combined_df.columns:
    date_index = filtered_combined_df.columns.get_loc(DATE_VALIDATE)
else:
    raise ValueError(f"Date {DATE_VALIDATE} not found in the columns of the dataframe")

# Prepare training data
Y_train_PCA = filtered_combined_df.iloc[:, :date_index]
REGRESSION_STEP = 12
Y_train_other = Y_train_PCA.iloc[:, REGRESSION_STEP:]
Y_reg_train = filtered_combined_df.iloc[:, :date_index + 1 - REGRESSION_STEP]

# Inspect the splits
print("Shapes of the splits:")
print("Y_train_PCA:", Y_train_PCA.shape)
print("Y_train_other:", Y_train_other.shape)
print("Y_reg_train:", Y_reg_train.shape)

Y_train_other_std = standardize(Y_train_other.values.T).T
Y_reg_train_std = standardize(Y_reg_train.values.T).T

# Fit the Dynamic Factor Model
model.std_data = Y_train_other_std.T  # Ensure the same subset of data is used for PCA
model.apply_pca()
model.yw_estimation()

# Prepare training and validation data for ElasticNet
train_split_index = int(model.factors.shape[1] * 0.8)

data_train = Y_train_other_std[:, :train_split_index].T
fac_train = model.factors[:, :train_split_index].T

data_validate = Y_train_other_std[:, train_split_index:].T
fac_validate = model.factors[:, train_split_index:].T

B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

# Validate model
y_hat_validate = model.enet_predict(fac_validate)

# Calculate RMSE for validation data using only the original 66 variables
rmse_value = RMSE(data_validate[:, :66], y_hat_validate[:, :66])

# Ensure variable names match RMSE values length
valid_variable_names = variable_names[:len(rmse_value)]

# Create a DataFrame with RMSE values and variable names
rmse_table = pd.DataFrame({'Variable': valid_variable_names, 'RMSE': rmse_value})
print(rmse_table)

# Print additional results
print(f"R2 in-sample: {r2_insample}")
print(f"ElasticNet intercept: {intercept}")

# Confirm the script has finished
print("Script execution completed.")

rmse_table.to_excel('rmse_combined.xlsx', index=False)