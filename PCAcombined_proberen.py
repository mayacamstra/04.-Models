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

# Separate static variables for RMSE calculation
static_df = combined_df.loc[combined_df.index[:66]]

# Save variable names
variable_names = static_df.index.tolist()

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
    model = DynamicFactorModel(Y_train, num_factors)
    
    # Fit the Dynamic Factor Model and apply PCA
    model.std_data = Y_train_std.T
    model.apply_pca()

    # Estimate the Yule-Walker equations
    model.yw_estimation()

    # Use 80% of the training data for training and 20% for testing
    train_split_index = int(model.factors.shape[1] * 0.8)
    data_train = Y_train_std[:, :train_split_index].T  # 80% of training data
    fac_train = model.factors[:, :train_split_index].T
    data_test = Y_train_std[:, train_split_index:].T  # 20% of training data
    fac_test = model.factors[:, train_split_index:].T
    
    # Fit ElasticNet model
    print("Fitting ElasticNet model...")
    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)

    # Validate model on out-of-sample data
    y_hat_test = model.enet_predict(fac_test)
    
    # Calculate residuals
    residuals_train = data_train - y_hat_train
    residuals_test = data_test - y_hat_test

    # Save the trained ElasticNet model for future predictions
    elastic_net_model = model.model_ena
    
    # Voorspel factoren en variabelen voor t+1
    current_train_data = Y_train_std  # Start met de trainingsdata
    next_timestamp = DATE_TRAIN_END + 1
    next_timestamp_str = next_timestamp.strftime('%Y-%m')

    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=Y_train.columns
    )
    
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    model.model_ena = elastic_net_model

    factor_forecast = model.factor_forecast(next_timestamp_str, scenarios=1)
    predicted_variables_t1 = model.enet_predict(factor_forecast.reshape(1, -1))

    # Store the predicted factors and variables for t+1
    if num_factors not in predicted_factors_dict:
        predicted_factors_dict[num_factors] = factor_forecast.T
    else:
        predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast.T))
    
    if num_factors not in predicted_variables_dict:
        predicted_variables_dict[num_factors] = predicted_variables_t1.T
    else:
        predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t1.T))

    current_train_data = np.hstack((current_train_data, predicted_variables_t1.T))

    # Herhaal voor t+2
    next_timestamp_2 = next_timestamp + 1
    next_timestamp_2_str = next_timestamp_2.strftime('%Y-%m')

    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp_str]
    )
    
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    model.model_ena = elastic_net_model

    factor_forecast_2 = model.factor_forecast(next_timestamp_2_str, scenarios=1)
    predicted_variables_t2 = model.enet_predict(factor_forecast_2.reshape(1, -1))

    # Store the predicted factors and variables for t+2
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_2.T))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t2.T))

    current_train_data = np.hstack((current_train_data, predicted_variables_t2.T))

    # Herhaal voor t+3
    next_timestamp_3 = next_timestamp_2 + 1
    next_timestamp_3_str = next_timestamp_3.strftime('%Y-%m')

    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp_str, next_timestamp_2_str]
    )
    
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    model.model_ena = elastic_net_model

    factor_forecast_3 = model.factor_forecast(next_timestamp_3_str, scenarios=1)
    predicted_variables_t3 = model.enet_predict(factor_forecast_3.reshape(1, -1))

    # Store the predicted factors and variables for t+3
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_3.T))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t3.T))

    current_train_data = np.hstack((current_train_data, predicted_variables_t3.T))

    # Herhaal voor t+4
    next_timestamp_4 = next_timestamp_3 + 1
    next_timestamp_4_str = next_timestamp_4.strftime('%Y-%m')

    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp_str, next_timestamp_2_str, next_timestamp_3_str]
    )
    
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    model.model_ena = elastic_net_model

    factor_forecast_4 = model.factor_forecast(next_timestamp_4_str, scenarios=1)
    predicted_variables_t4 = model.enet_predict(factor_forecast_4.reshape(1, -1))

    # Store the predicted factors and variables for t+4
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_4.T))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t4.T))

    current_train_data = np.hstack((current_train_data, predicted_variables_t4.T))

    # Herhaal voor t+5
    next_timestamp_5 = next_timestamp_4 + 1
    next_timestamp_5_str = next_timestamp_5.strftime('%Y-%m')

    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp_str, next_timestamp_2_str, next_timestamp_3_str, next_timestamp_4_str]
    )
    
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    model.model_ena = elastic_net_model

    factor_forecast_5 = model.factor_forecast(next_timestamp_5_str, scenarios=1)
    predicted_variables_t5 = model.enet_predict(factor_forecast_5.reshape(1, -1))

    # Store the predicted factors and variables for t+5
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_5.T))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t5.T))

    current_train_data = np.hstack((current_train_data, predicted_variables_t5.T))

    # Nu kun je verdergaan met het berekenen van RMSE en andere metrics zoals eerder gedaan
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
results_df.to_excel('results_PCAcombined_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx', index=False)
# Sla de voorspelde matrices op als Excel-bestanden voor elk aantal factoren
for num_factors, matrix in predicted_factors_dict.items():
    pd.DataFrame(matrix).to_excel(f'PCAcombined_predicted_factors_matrix_{num_factors}.xlsx', index=False)
    
for num_factors, matrix in predicted_variables_dict.items():
    pd.DataFrame(matrix).to_excel(f'PCAcombined_predicted_variables_matrix_{num_factors}.xlsx', index=False)
# Print feedback naar de gebruiker
print("Results saved to results_PCAcombined_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx")
print("Predicted factors and variables matrices saved to separate Excel files for each number of factors.")
