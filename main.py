import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from utils import standardize, RMSE
from factor_model import DynamicFactorModel
from datetime import datetime

# Load data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
df_data = load_data(file_path)

# Define number of factors
num_factors = 9

# Initialize the model
model = DynamicFactorModel(df_data, num_factors)

# Define validation date and split the data
DATE_VALIDATE = pd.Period('2010-01', freq='M')
if DATE_VALIDATE in df_data.columns:
    date_index = df_data.columns.get_loc(DATE_VALIDATE)
else:
    raise ValueError(f"Date {DATE_VALIDATE} not found in the columns of the dataframe")

# Prepare training data
Y_train_PCA = df_data.iloc[:, :date_index]
REGRESSION_STEP = 12
Y_train_other = Y_train_PCA.iloc[REGRESSION_STEP:, :]
Y_reg_train = df_data.iloc[:, :date_index + 1 - REGRESSION_STEP]

Y_train_other_std = standardize(Y_train_other.values.T).T
Y_reg_train_std = standardize(Y_reg_train.values.T).T

# Fit the Dynamic Factor Model
model.std_data = Y_train_other_std.T  # Ensure the same subset of data is used for PCA
model.apply_pca()
model.yw_estimation()

# Prepare training and validation data for ElasticNet
data_train = Y_train_other_std[:, :int(Y_train_other_std.shape[1] * 0.8)].T
fac_train = model.factors[:, :int(model.factors.shape[1] * 0.8)].T

data_validate = Y_train_other_std[:, int(Y_train_other_std.shape[1] * 0.8):].T
fac_validate = model.factors[:, int(model.factors.shape[1] * 0.8):].T

B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

# Validate model
y_hat_validate = model.enet_predict(fac_validate)

# Calculate RMSE for validation data
rmse_value = RMSE(data_validate, y_hat_validate)

# Plot RMSE values
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(rmse_value)), rmse_value)
# plt.xlabel('Variable Index')
# plt.ylabel('RMSE')
# plt.title('RMSE for Each Variable in the Validation Set')
# plt.show()

# Print RMSE values as a table
rmse_table = pd.DataFrame({'Variable Index': range(len(rmse_value)), 'RMSE': rmse_value})
print(rmse_table)

# Print results
print(f"RMSE: {rmse_value}")
print(f"R2 in-sample: {r2_insample}")
print(f"ElasticNet intercept: {intercept}")
# Confirm the script has finished
print("Script execution completed.")