import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_combined_data, filter_data
from utils import standardize, RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model import DynamicFactorModel

# Zorg ervoor dat de directory bestaat waar we de plots gaan opslaan
plot_dir = "plots_PLScombined"
os.makedirs(plot_dir, exist_ok=True)

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

# Initialize a list to store results
results = []

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

    # Fit ElasticNet model
    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)

    # Validate model on out-of-sample data
    y_hat_validate = model.enet_predict(fac_validate)
    
    # Calculate residuals
    residuals_train = data_train - y_hat_train
    residuals_validate = data_validate - y_hat_validate

    # Calculate RMSE and R² for validation data using only the original 66 variables
    rmse_value_in_sample = RMSE(data_train[:, :66], y_hat_train[:, :66])
    rmse_value_out_sample = RMSE(data_validate[:, :66], y_hat_validate[:, :66])
    r2_out_sample = calculate_r2(data_validate[:, :66], y_hat_validate[:, :66])

    # Calculate log-likelihood, AIC, and BIC
    log_like_value = log_likelihood(data_train[:, :66], y_hat_train[:, :66])
    aic_value, bic_value = calculate_aic_bic(y_hat_train, data_train, num_factors)

    # Calculate adjusted R²
    adj_r2_in_sample = adjusted_r2(r2_insample, data_train.shape[0], num_factors)
    adj_r2_out_sample = adjusted_r2(r2_out_sample, data_validate.shape[0], num_factors)

    # Average RMSE values across variables
    avg_rmse_in_sample = rmse_value_in_sample.mean()
    avg_rmse_out_sample = rmse_value_out_sample.mean()

    # Append the results to the list
    results.append({
        'Num_Factors': num_factors,
        'RMSE_InSample': avg_rmse_in_sample,
        'R2_InSample': r2_insample,
        'Adjusted_R2_InSample': adj_r2_in_sample,
        'RMSE_OutSample': avg_rmse_out_sample,
        'R2_OutSample': r2_out_sample,
        'Adjusted_R2_OutSample': adj_r2_out_sample,
        'Log_Likelihood': log_like_value,
        'AIC': aic_value,
        'BIC': bic_value
    })

    # Plot residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_hat_train.flatten(), residuals_train.flatten())
    plt.title(f'Residuals vs Fitted (In-sample) - {num_factors} Factors')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--')

    plt.subplot(1, 2, 2)
    plt.scatter(y_hat_validate.flatten(), residuals_validate.flatten())
    plt.title(f'Residuals vs Fitted (Out-of-sample) - {num_factors} Factors')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--')

    plt.tight_layout()
    
    # Save the plot instead of showing it
    plt.savefig(f"{plot_dir}/residuals_{num_factors}_factors.png")
    plt.close()  # Close the figure to free up memory

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('results_PLScombined_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx', index=False)

print("Results saved to results_PLScombined_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx")