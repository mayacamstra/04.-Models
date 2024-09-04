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

    # Debugging: Check if the ElasticNet model is set
    if model.model_ena is None:
        print("Error: ElasticNet model is not set after fitting. Check enet_fit method.")
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    else:
        print("ElasticNet model trained successfully.")

    # Debugging: Print model coefficients and intercept
    print(f"ElasticNet coefficients: {B_matrix}")
    print(f"ElasticNet intercept: {intercept}")

    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)

    # Validate model on out-of-sample data
    y_hat_test = model.enet_predict(fac_test)
    
    # Calculate residuals
    residuals_train = data_train - y_hat_train
    residuals_test = data_test - y_hat_test
    
    # Debugging: Print residuals
    print(f"Residuals (in-sample): {residuals_train}")
    print(f"Residuals (out-sample): {residuals_test}")

    # Houd het getrainde ElasticNet model vast voor toekomstige voorspellingen
    elastic_net_model = model.model_ena  # Save the trained ElasticNet model
    
    # Voorspel factoren en variabelen voor t+1
    current_train_data = Y_train_std  # Start met de trainingsdata
    next_timestamp = DATE_TRAIN_END + 1
    next_timestamp_str = next_timestamp.strftime('%Y-%m')

    # Gebruik de laatste beschikbare training data
    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=Y_train.columns
    )
    
    # Initialiseer model met de DataFrame
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    
    # Hergebruik het getrainde ElasticNet model
    model.model_ena = elastic_net_model

    # Debugging: Check if model_ena is set before predicting
    if model.model_ena is None:
        print("Error: ElasticNet model is not set before predicting t+1.")
        raise ValueError("ElasticNet model is not set before predicting t+1.")

    # Voorspel factoren voor t+1
    factor_forecast = model.factor_forecast(next_timestamp_str, scenarios=1)
    if factor_forecast.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast.shape[1]} features")

    # Debugging: Print factor_forecast values
    print(f"Factor forecast for t+1: {factor_forecast}")
    
    # Predict the static and forward-looking variables based on the forecasted factors
    predicted_variables_t1 = model.enet_predict(factor_forecast.reshape(1, -1))
    print(f"Predicted variables for t+1: {predicted_variables_t1}")

    # Voeg de voorspelde variabelen toe aan de dataset
    current_train_data = np.hstack((current_train_data, predicted_variables_t1.T))

    # Herhaal voor t+2
    next_timestamp_2 = next_timestamp + 1
    next_timestamp_2_str = next_timestamp_2.strftime('%Y-%m')
    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp]
    )
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    
    # Hergebruik het getrainde ElasticNet model
    model.model_ena = elastic_net_model

    # Debugging: Check if model_ena is set before predicting t+2
    if model.model_ena is None:
        print("Error: ElasticNet model is not set before predicting t+2.")
        raise ValueError("ElasticNet model is not set before predicting t+2.")

    factor_forecast_2 = model.factor_forecast(next_timestamp_2_str, scenarios=1)
    predicted_variables_t2 = model.enet_predict(factor_forecast_2.reshape(1, -1))
    print(f"Predicted variables for t+2: {predicted_variables_t2}")

    current_train_data = np.hstack((current_train_data, predicted_variables_t2.T))

    # Herhaal voor t+3
    next_timestamp_3 = next_timestamp_2 + 1
    next_timestamp_3_str = next_timestamp_3.strftime('%Y-%m')
    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp, next_timestamp_2]
    )
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    
    # Hergebruik het getrainde ElasticNet model
    model.model_ena = elastic_net_model

    # Debugging: Check if model_ena is set before predicting t+3
    if model.model_ena is None:
        print("Error: ElasticNet model is not set before predicting t+3.")
        raise ValueError("ElasticNet model is not set before predicting t+3.")

    factor_forecast_3 = model.factor_forecast(next_timestamp_3_str, scenarios=1)
    predicted_variables_t3 = model.enet_predict(factor_forecast_3.reshape(1, -1))
    print(f"Predicted variables for t+3: {predicted_variables_t3}")

    current_train_data = np.hstack((current_train_data, predicted_variables_t3.T))

    # Herhaal voor t+4
    next_timestamp_4 = next_timestamp_3 + 1
    next_timestamp_4_str = next_timestamp_4.strftime('%Y-%m')
    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp, next_timestamp_2, next_timestamp_3]
    )
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    
    # Hergebruik het getrainde ElasticNet model
    model.model_ena = elastic_net_model

    # Debugging: Check if model_ena is set before predicting t+4
    if model.model_ena is None:
        print("Error: ElasticNet model is not set before predicting t+4.")
        raise ValueError("ElasticNet model is not set before predicting t+4.")

    factor_forecast_4 = model.factor_forecast(next_timestamp_4_str, scenarios=1)
    predicted_variables_t4 = model.enet_predict(factor_forecast_4.reshape(1, -1))
    print(f"Predicted variables for t+4: {predicted_variables_t4}")

    current_train_data = np.hstack((current_train_data, predicted_variables_t4.T))

    # Herhaal voor t+5
    next_timestamp_5 = next_timestamp_4 + 1
    next_timestamp_5_str = next_timestamp_5.strftime('%Y-%m')
    current_train_data_df = pd.DataFrame(
        current_train_data,
        index=Y_train.index,
        columns=list(Y_train.columns) + [next_timestamp, next_timestamp_2, next_timestamp_3, next_timestamp_4]
    )
    model = DynamicFactorModel(current_train_data_df, num_factors)
    model.std_data = current_train_data_df.values.T
    model.apply_pca()
    model.yw_estimation()
    
    # Hergebruik het getrainde ElasticNet model
    model.model_ena = elastic_net_model

    # Debugging: Check if model_ena is set before predicting t+5
    if model.model_ena is None:
        print("Error: ElasticNet model is not set before predicting t+5.")
        raise ValueError("ElasticNet model is not set before predicting t+5.")

    factor_forecast_5 = model.factor_forecast(next_timestamp_5_str, scenarios=1)
    predicted_variables_t5 = model.enet_predict(factor_forecast_5.reshape(1, -1))
    print(f"Predicted variables for t+5: {predicted_variables_t5}")

    current_train_data = np.hstack((current_train_data, predicted_variables_t5.T))

    # Nu kun je verdergaan met het berekenen van RMSE en andere metrics zoals eerder gedaan

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
