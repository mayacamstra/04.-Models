import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, filter_data
from utils import standardize, RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model import DynamicFactorModel

# Zorg ervoor dat de directory bestaat waar we de plots gaan opslaan
plot_dir = "plots_PCAstatic"
os.makedirs(plot_dir, exist_ok=True)

# Load and filter data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
df_data = load_data(file_path)
filtered_df = filter_data(df_data)

# Save variable names
variable_names = filtered_df.index.tolist()

# Define training and validation periods and split the data
DATE_TRAIN_END = pd.Period('2019-12', freq='M')
DATE_VALIDATE_START = pd.Period('2020-01', freq='M')
DATE_VALIDATE_END = pd.Period('2023-11', freq='M')

# Split the data into training and validation sets
Y_train = filtered_df.loc[:, :DATE_TRAIN_END]  # Data until 2019-12
Y_validate = filtered_df.loc[:, DATE_VALIDATE_START:DATE_VALIDATE_END]  # Data from 2020-01 to 2023-11

# Standardize the datasets without transposing
Y_train_std = standardize(Y_train.values)  # Keep shape as (66, 300)
Y_validate_std = standardize(Y_validate.values)  # Keep shape as (66, 47)

# Debug: check shapes of standardized data
print(f"Shape of Y_train_std: {Y_train_std.shape} (expected: 66 variables x 300 months)")
print(f"Shape of Y_validate_std: {Y_validate_std.shape} (expected: 66 variables x 47 months)")

# Define the range of factors to test
factor_range = range(5, 13)  # Example range from 5 to 12 factors

# Initialize a list to store results
results = []

# Initialize dictionaries to store predicted factors and variables matrices per number of factors
predicted_factors_dict = {}
predicted_variables_dict = {}

for num_factors in factor_range:
    print(f"\nEvaluating model with {num_factors} factors")

    # Initialize the dynamic factor model
    model = DynamicFactorModel(Y_train, num_factors)  # Use only the training set here

    # Fit the Dynamic Factor Model and apply PCA without transposing
    model.std_data = Y_train_std  # Keep original shape (66, 300)
    model.apply_pca()  # Apply PCA to extract factors from the variables

    # Debug: Print the shape of extracted factors
    print(f"Shape of extracted PCA factors: {model.factors.shape} (expected: {num_factors} factors x 300 months)")

    # Estimate the Yule-Walker equations
    model.yw_estimation()

    # Use 80% of the training data for training and 20% for testing
    train_split_index = int(model.factors.shape[1] * 0.8)
    data_train = Y_train_std[:, :train_split_index].T  # 80% of training data
    fac_train = model.factors[:, :train_split_index].T
    data_test = Y_train_std[:, train_split_index:].T  # 20% of training data
    fac_test = model.factors[:, train_split_index:].T

    # Debugging: Check shapes of training and test datasets
    print(f"Shape of data_train: {data_train.shape} (expected: 240 months x 66 variables)")
    print(f"Shape of fac_train: {fac_train.shape} (expected: 240 months x {num_factors} factors)")
    print(f"Shape of data_test: {data_test.shape} (expected: 60 months x 66 variables)")
    print(f"Shape of fac_test: {fac_test.shape} (expected: 60 months x {num_factors} factors)")

    # Fit ElasticNet model
    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)

    # Validate model on out-of-sample data
    y_hat_test = model.enet_predict(fac_test)

    # Calculate residuals
    residuals_train = data_train - y_hat_train
    residuals_test = data_test - y_hat_test

    # Debugging: Check shape of residuals
    print(f"Residuals train shape: {residuals_train.shape} (expected: 240 months x 66 variables)")
    print(f"Residuals test shape: {residuals_test.shape} (expected: 60 months x 66 variables)")

    # Predict factors for the next timestamp after the training set (2020-01)
    next_timestamp = '2020-01'
    factor_forecast = model.factor_forecast(next_timestamp, scenarios=1)

    # Debugging: Check shape of predicted factors for 2020-01
    print(f"Shape of factor_forecast for {next_timestamp}: {factor_forecast.shape} (expected: 1 month x {num_factors} factors)")

    # Check if the predicted factors have the correct shape
    if factor_forecast.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast.shape[1]} features")

    # Predict the original variables based on the forecasted factors
    predicted_variables_t1 = model.enet_predict(factor_forecast.reshape(1, -1))

    # Debugging: Check shape of predicted variables for 2020-01
    print(f"Predicted variables shape for {next_timestamp}: {predicted_variables_t1.shape} (expected: 1 month x 66 variables)")

    # Append the results for factors and variables to dictionaries
    predicted_factors_dict[num_factors] = factor_forecast.T
    predicted_variables_dict[num_factors] = predicted_variables_t1.T

    # Append results, plot residuals, etc.
    # Calculate RMSE and R² for in-sample and test data
    rmse_value_in_sample = RMSE(data_train, y_hat_train)
    rmse_value_test_sample = RMSE(data_test, y_hat_test)
    r2_test_sample = calculate_r2(data_test, y_hat_test)

    # Calculate log-likelihood, AIC, and BIC
    log_like_value = log_likelihood(data_train, y_hat_train)
    aic_value, bic_value = calculate_aic_bic(y_hat_train, data_train, num_factors)

    # Calculate adjusted R²
    adj_r2_in_sample = adjusted_r2(r2_insample, data_train.shape[0], num_factors)
    adj_r2_test_sample = adjusted_r2(r2_test_sample, data_test.shape[0], num_factors)

    # Average RMSE values across variables
    avg_rmse_in_sample = rmse_value_in_sample.mean()
    avg_rmse_test_sample = rmse_value_test_sample.mean()

    # Append the results to the list
    results.append({
        'Num_Factors': num_factors,
        'RMSE_InSample': avg_rmse_in_sample,
        'R2_InSample': r2_insample,
        'Adjusted_R2_InSample': adj_r2_in_sample,
        'RMSE_TestSample': avg_rmse_test_sample,
        'R2_TestSample': r2_test_sample,
        'Adjusted_R2_TestSample': adj_r2_test_sample,
        'Log_Likelihood': log_like_value,
        'AIC': aic_value,
        'BIC': bic_value
    })

    # Plot residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_hat_train.flatten(), residuals_train.flatten())
    plt.title(f'Residuals vs Fitted (In-sample Train) - {num_factors} Factors')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--')

    plt.subplot(1, 2, 2)
    plt.scatter(y_hat_test.flatten(), residuals_test.flatten())
    plt.title(f'Residuals vs Fitted (In-sample Test) - {num_factors} Factors')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()

    # Save the plot instead of showing it
    plt.savefig(f"{plot_dir}/residuals_{num_factors}_factors.png")
    plt.close()  # Close the figure to free up memory

# Converteer de resultatenlijsten naar DataFrames voor eventuele verdere analyse of opslag
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel('TESTresults_PCAstatic_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx', index=False)

# Sla de voorspelde matrices op als Excel-bestanden voor elk aantal factoren
for num_factors, matrix in predicted_factors_dict.items():
    pd.DataFrame(matrix).to_excel(f'TESTpredicted_factors_matrix_{num_factors}.xlsx', index=False)

for num_factors, matrix in predicted_variables_dict.items():
    pd.DataFrame(matrix).to_excel(f'TESTpredicted_variables_matrix_{num_factors}.xlsx', index=False)

# Print feedback naar de gebruiker
print("Results saved to results_PCAstatic_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx")
print("Predicted factors and variables matrices saved to separate Excel files for each number of factors.")
