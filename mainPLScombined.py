import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_combined_data, filter_data
from utils import standardize, RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model import DynamicFactorModel  # Gebruik de bestaande DynamicFactorModel klasse

# Zorg ervoor dat de directory bestaat waar we de plots gaan opslaan
plot_dir = "plots_PLScombined"
os.makedirs(plot_dir, exist_ok=True)

# Load and filter combined data (Static + Forward)
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
Y_train_std = standardize(Y_train.values)
Y_validate_std = standardize(Y_validate.values)

# Debug: check shapes of standardized data
print(f"Shape of Y_train_std: {Y_train_std.shape} (expected: variables x months)")
print(f"Shape of Y_validate_std: {Y_validate_std.shape} (expected: variables x months)")

# Define the range of factors to test
factor_range = range(5, 13)  # Example range from 5 to 12 factors

# Initialize a list to store results
results = []

# Initialize dictionaries to store predicted factors and variables matrices per number of factors
predicted_factors_dict = {}
predicted_variables_dict = {}

for num_factors in factor_range:
    print(f"\nEvaluating model with {num_factors} factors using PLS method")

    # Initialize the dynamic factor model with the PLS method
    model = DynamicFactorModel(Y_train, num_factors, method='PLS')  # Specify PLS as the method

    # Transpose so PLS works on the variables (variables x time points)
    model.std_data = Y_train_std.T  # Transpose to shape (time points, variables)
    model.apply_pls(Y_train_std.T, Y_train_std.T)  # PLS should be applied on (time points, variables)

    # Debug: Print the shape of factors after PLS
    print(f"Shape of PLS factors: {model.factors.shape}")

    # Ensure factors are 2D (num_factors, num_time_points)
    if len(model.factors.shape) == 1:
        raise ValueError(f"PLS factors are not 2D. Got shape {model.factors.shape} instead of expected 2D array.")

    # Transpose factors if necessary
    if model.factors.shape[0] != num_factors:
        model.factors = model.factors.T

    print(f"Shape of factors after adjustment: {model.factors.shape}")

    model.yw_estimation()

    # Debug: Print the shape of factors after YW estimation
    factors = model.factors
    print(f"Shape of factors after PLS: {factors.shape}")

    # Use 80% of the training data for training and 20% for testing
    train_split_index = int(factors.shape[1] * 0.8)
    data_train = Y_train_std[:, :train_split_index].T  # 80% of training data
    fac_train = factors[:, :train_split_index].T
    data_test = Y_train_std[:, train_split_index:].T  # 20% of training data
    fac_test = factors[:, train_split_index:].T

    # Debug: Print the shapes of training and test sets
    print(f"Shape of data_train: {data_train.shape}, fac_train: {fac_train.shape}")
    print(f"Shape of data_test: {data_test.shape}, fac_test: {fac_test.shape}")

    # Fit ElasticNet model
    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)

    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)

    # Validate model on out-of-sample data
    y_hat_test = model.enet_predict(fac_test)

    # Calculate residuals
    residuals_train = data_train - y_hat_train
    residuals_test = data_test - y_hat_test

    # Debug: Print residuals shape
    print(f"Residuals train shape: {residuals_train.shape}")
    print(f"Residuals test shape: {residuals_test.shape}")

    # Initialiseer variabelen voor voorspellingen
    current_train_data = Y_train_std
    current_index = list(Y_train.columns)

    # Voorspellingen voor tijdstappen t+1 tot t+48
    for t in range(1, 48):
        next_timestamp = current_index[-1] + 1  # Bereken volgende timestamp
        next_timestamp_str = next_timestamp.strftime('%Y-%m')

        # Voorspel de volgende set factoren
        factor_forecast = model.factor_forecast(next_timestamp_str, scenarios=1)

        # Debug: Print the shape of factor_forecast
        print(f"Shape of factor_forecast for t+{t}: {factor_forecast.shape}")

        # Controleer de vorm van de voorspelde factoren
        if factor_forecast.shape[1] != num_factors:
            # Zorg ervoor dat je alleen het juiste aantal factoren behoudt
            factor_forecast = factor_forecast[:, :num_factors]
            print(f"Adjusted factor_forecast shape for t+{t}: {factor_forecast.shape}")

        # Controleer nogmaals of de aangepaste voorspelling klopt
        if factor_forecast.shape[1] != num_factors:
            raise ValueError(f"Expected {num_factors} features, got {factor_forecast.shape[1]} features")

        # Voeg de voorspelde factoren toe aan de matrix in de dictionary
        predicted_factors_dict[num_factors] = np.hstack(
            (predicted_factors_dict.get(num_factors, np.empty((num_factors, 0))), factor_forecast.T))

        # Voorspel de originele variabelen op basis van de voorspelde factoren
        predicted_variables = model.enet_predict(factor_forecast)

        # Voeg de voorspelde variabelen toe aan de matrix in de dictionary
        predicted_variables_dict[num_factors] = np.hstack(
            (predicted_variables_dict.get(num_factors, np.empty((Y_train_std.shape[0], 0))), predicted_variables.T))

        # Voeg de voorspelde waarden voor de huidige stap toe aan de trainingsdata
        extended_train_data = np.hstack((current_train_data, predicted_variables.T))

        # Standaardiseer opnieuw de uitgebreide dataset
        extended_train_data_std = standardize(extended_train_data.T).T

        # Debug: Print the shape of extended_train_data_std
        print(f"Shape of extended_train_data_std for t+{t}: {extended_train_data_std.shape}")

        # Update de index met een nieuwe tijdstempel
        extended_index = current_index + [next_timestamp]

        # Zet de uitgebreide dataset om naar een pandas DataFrame met een correcte index
        extended_train_df = pd.DataFrame(extended_train_data_std, index=Y_train.index, columns=extended_index)

        # Fit het model opnieuw met de uitgebreide trainingsset
        model = DynamicFactorModel(extended_train_df, num_factors, method='PLS')
        model.std_data = extended_train_data_std.T

        # Let op: altijd transponeren zodat de vorm correct blijft
        model.apply_pls(extended_train_data_std.T, extended_train_data_std.T)

        # Controleer of de factoren correct zijn getransformeerd (altijd 2D, num_factors x time_points)
        if model.factors.shape[0] != num_factors:
            model.factors = model.factors.T  # Transponeer indien nodig
        print(f"Shape of factors after adjustment: {model.factors.shape}")

        model.yw_estimation()

        # Hertraining van ElasticNet model met de uitgebreide trainingsdata
        fac_train_extended = model.factors.T
        data_train_extended = extended_train_data_std.T

        # Simpele print statement
        print(f"Training extended model for t+{t+1} with data and factors...")

        # Fit ElasticNet model met de uitgebreide trainingsdata
        model.enet_fit(data_train_extended, fac_train_extended)

        # Update current data voor de volgende iteratie
        current_train_data = extended_train_data_std
        current_index = extended_index

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
results_df.to_excel('results_PLScombined_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx', index=False)

# Sla de voorspelde matrices op als Excel-bestanden voor elk aantal factoren
for num_factors, matrix in predicted_factors_dict.items():
    pd.DataFrame(matrix).to_excel(f'PLScombined_predicted_factors_matrix_{num_factors}.xlsx', index=False)

for num_factors, matrix in predicted_variables_dict.items():
    pd.DataFrame(matrix).to_excel(f'PLScombined_predicted_variables_matrix_{num_factors}.xlsx', index=False)

print("Results saved to results_PLScombined_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx")
print("Predicted factors and variables matrices saved to separate Excel files for each number of factors.")
