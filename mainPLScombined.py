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

# Save variable names
variable_names = static_df.index.tolist()

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

# Define the range of factors to test
factor_range = range(5, 13)  # Range from 5 to 12 factors

for num_factors in factor_range:
    print(f"\nEvaluating model with {num_factors} factors")

    # Initialize the dynamic factor model with PLS method
    model = DynamicFactorModel(filtered_combined_df, num_factors, method='PLS')

    # Separate X and Y for PLS
    X_train = Y_train_std
    Y_train = Y_train_std[:66, :]  # Only the first 66 variables are used as Y

    # Fit the Dynamic Factor Model and apply PLS
    model.std_data = X_train.T
    model.apply_pls(X_train.T, Y_train.T)  # Apply PLS with full X and only first 66 Y

    # Transpose the factors to match the expected shape
    model.factors = model.factors.T  # Transpose to get (num_factors, 300)

    # Prepare training and validation data for ElasticNet
    train_split_index = int(model.factors.shape[1] * 0.8)

    data_train = Y_train_std[:, :train_split_index].T
    fac_train = model.factors[:, :train_split_index].T

    data_validate = Y_validate_std.T
    fac_validate = model.factors[:, train_split_index:train_split_index + 47].T

    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

    # Validate model
    y_hat_validate = model.enet_predict(fac_validate)

    # Calculate RMSE for validation data using only the original 66 variables
    try:
        rmse_value = RMSE(data_validate[:, :66], y_hat_validate[:, :66])
        # Ensure variable names match RMSE values length
        valid_variable_names = variable_names[:len(rmse_value)]
        # Print RMSE values
        print(f"RMSE for {num_factors} factors:")
        print(pd.DataFrame({'Variable': valid_variable_names, 'RMSE': rmse_value}))
    except ValueError as e:
        print(f"RMSE calculation error: {e}")
        print(f"Shape mismatch details - data_validate: {data_validate.shape}, y_hat_validate: {y_hat_validate.shape}")

    # Print additional results
    print(f"R2 in-sample for {num_factors} factors: {r2_insample}")

# Confirm the script has finished
print("Script execution completed.")
