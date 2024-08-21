import pandas as pd
from data_loader import load_data, filter_data
from utils import standardize, RMSE, calculate_r2
from factor_model import DynamicFactorModel

# Load and filter data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
df_data = load_data(file_path)

# Apply Christiano-Fitzgerald filter
filtered_df = filter_data(df_data)

# Save variable names
variable_names = filtered_df.index.tolist()

# Define validation date and split the data
DATE_VALIDATE = pd.Period('2020-01', freq='M')
if DATE_VALIDATE in filtered_df.columns:
    date_index = filtered_df.columns.get_loc(DATE_VALIDATE)
else:
    raise ValueError(f"Date {DATE_VALIDATE} not found in the columns of the dataframe")

# Prepare training data until 2019-12, Validation from 2020-01 to 2023-11 (47 months)
Y_train_PLS = filtered_df.iloc[:, :date_index]
Y_validate = filtered_df.iloc[:, date_index:date_index + 47]

# Standardize the datasets
Y_train_std = standardize(Y_train_PLS.values.T).T
Y_validate_std = standardize(Y_validate.values.T).T

# Define the range of factors to test
factor_range = range(5, 13)  # Range from 5 to 12 factors

for num_factors in factor_range:
    print(f"\nEvaluating model with {num_factors} factors")

    # Initialize the dynamic factor model with PLS method
    model = DynamicFactorModel(filtered_df, num_factors, method='PLS')

    # Fit the Dynamic Factor Model and apply PLS
    model.std_data = Y_train_std.T
    model.apply_pls(Y_train_std.T, Y_train_std.T)

    # Transpose the factors to match the expected shape
    model.factors = model.factors.T  # Transpose to get (num_factors, 300)

    # Prepare training and validation data for ElasticNet
    train_split_index = int(model.factors.shape[1] * 0.8)

    data_train = Y_train_std[:, :train_split_index].T
    fac_train = model.factors[:, :train_split_index].T

    data_validate = Y_validate_std.T
    fac_validate = model.factors[:, train_split_index:train_split_index + 47].T

    # Fit ElasticNet model
    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)

    # Validate model on out-of-sample data
    y_hat_validate = model.enet_predict(fac_validate)

    # Calculate RMSE and R² for in-sample and out-of-sample data
    try:
        rmse_value_in_sample = RMSE(data_train, y_hat_train)
        rmse_value_out_sample = RMSE(data_validate, y_hat_validate)
        r2_out_sample = calculate_r2(data_validate, y_hat_validate)

        # Ensure variable names match RMSE values length
        valid_variable_names = variable_names[:len(rmse_value_out_sample)]

        # Print RMSE and R² values
        print(f"RMSE (in-sample) for {num_factors} factors:")
        print(pd.DataFrame({'Variable': valid_variable_names, 'RMSE': rmse_value_in_sample}))
        print(f"R2 (in-sample) for {num_factors} factors: {r2_insample}")

        print(f"RMSE (out-of-sample) for {num_factors} factors:")
        print(pd.DataFrame({'Variable': valid_variable_names, 'RMSE': rmse_value_out_sample}))
        print(f"R2 (out-of-sample) for {num_factors} factors: {r2_out_sample}")

    except ValueError as e:
        print(f"RMSE calculation error: {e}")
        print(f"Shape mismatch details - data_validate: {data_validate.shape}, y_hat_validate: {y_hat_validate.shape}")

# Confirm the script has finished
print("Script execution completed.")
