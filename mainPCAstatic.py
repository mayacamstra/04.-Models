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
    model = DynamicFactorModel(Y_train, num_factors)  # Use only the training set here
    
    # Fit the Dynamic Factor Model and apply PCA
    model.std_data = Y_train_std.T
    model.apply_pca()  # Apply the simpler PCA method
    # Estimate the Yule-Walker equations
    model.yw_estimation()
    # Use 80% of the training data for training and 20% for testing
    train_split_index = int(model.factors.shape[1] * 0.8)
    data_train = Y_train_std[:, :train_split_index].T  # 80% of training data
    fac_train = model.factors[:, :train_split_index].T
    data_test = Y_train_std[:, train_split_index:].T  # 20% of training data
    fac_test = model.factors[:, train_split_index:].T
    # Fit ElasticNet model
    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)
    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)
    # Validate model on in-sample test data
    y_hat_test = model.enet_predict(fac_test)
    # Calculate residuals
    residuals_train = data_train - y_hat_train
    residuals_test = data_test - y_hat_test
    # Voorspel factoren voor de volgende tijdstempel na de laatste van de trainingsset
    next_timestamp = '2020-01'  # De volgende maand na de laatste trainingsmaand
    factor_forecast = model.factor_forecast(next_timestamp, scenarios=1)
    # Zorg ervoor dat de voorspelde factoren de juiste vorm hebben
    if factor_forecast.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast.shape[1]} features")
    # Voeg de voorspelde factoren toe aan de matrix in de dictionary
    if num_factors not in predicted_factors_dict:
        predicted_factors_dict[num_factors] = factor_forecast.T
    else:
        predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast.T))
    # Predict the original variables based on the forecasted factors
    predicted_variables_t1 = model.enet_predict(factor_forecast.reshape(1, -1))
    # Voeg de voorspelde variabelen toe aan de matrix in de dictionary
    if num_factors not in predicted_variables_dict:
        predicted_variables_dict[num_factors] = predicted_variables_t1.T
    else:
        predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t1.T))
    # print(f"Predicted variables for {next_timestamp}:\n", predicted_variables_t1)
    # Voeg de voorspelde waarden voor 't+1' toe aan de trainingsdata
    extended_train_data = np.hstack((Y_train_std, predicted_variables_t1.T))
    # Standaardiseer opnieuw de uitgebreide dataset
    extended_train_data_std = standardize(extended_train_data.T).T
    # Zorg ervoor dat de uitgebreide dataset een correcte PeriodIndex behoudt
    # Maak een nieuwe index met tijdstempels
    extended_index = list(Y_train.columns) + [pd.Period('2020-01', freq='M')]
    # Zet de uitgebreide dataset om naar een pandas DataFrame met een correcte index
    extended_train_df = pd.DataFrame(extended_train_data_std, index=Y_train.index, columns=extended_index)
    # Fit het model opnieuw met de uitgebreide trainingsset inclusief 't+1' voorspellingen
    model = DynamicFactorModel(extended_train_df, num_factors)
    model.std_data = extended_train_data_std.T
    model.apply_pca()
    model.yw_estimation()
    # Hertraining van ElasticNet model met de uitgebreide trainingsdata
    fac_train_extended = model.factors.T
    data_train_extended = extended_train_data_std.T
    # Debug: print output voor debugging
    print("Training extended model for t+2 with data and factors...")
    # Fit ElasticNet model met de uitgebreide trainingsdata
    model.enet_fit(data_train_extended, fac_train_extended)
    # Controleer of het model correct is ingesteld
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    # Gebruik alleen de factoren van 't+1' voor het voorspellen van 't+2'
    next_timestamp_2 = pd.Period(next_timestamp, freq='M') + 1
    next_timestamp_2_str = next_timestamp_2.strftime('%Y-%m')
    factor_forecast_2 = model.factor_forecast(next_timestamp_2_str, scenarios=1)
    # Zorg ervoor dat de vorm van factor_forecast_2 overeenkomt met de verwachte inputdimensie
    if factor_forecast_2.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_2.shape[1]} features")
    # Voeg de voorspelde factoren voor t+2 toe aan de matrix in de dictionary
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_2.T))
    # Voorspel de originele variabelen op basis van de voorspelde factoren van 't+1'
    predicted_variables_t2 = model.enet_predict(factor_forecast_2.reshape(1, -1))
    # Voeg de voorspelde variabelen voor t+2 toe aan de matrix in de dictionary
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t2.T))

    # print(f"Predicted variables for {next_timestamp_2_str}:\n", predicted_variables_t2)

    # Voeg de voorspelde waarden voor 't+2' toe aan de trainingsdata
    extended_train_data_2 = np.hstack((extended_train_data, predicted_variables_t2.T))

    # Standaardiseer opnieuw de uitgebreide dataset
    extended_train_data_2_std = standardize(extended_train_data_2.T).T

    # Update de index met een nieuwe tijdstempel
    extended_index_2 = extended_index + [next_timestamp_2]

    # Zet de uitgebreide dataset om naar een pandas DataFrame met een correcte index
    extended_train_df_2 = pd.DataFrame(extended_train_data_2_std, index=Y_train.index, columns=extended_index_2)

    # Fit het model opnieuw met de uitgebreide trainingsset inclusief 't+2' voorspellingen
    model = DynamicFactorModel(extended_train_df_2, num_factors)
    model.std_data = extended_train_data_2_std.T
    model.apply_pca()
    model.yw_estimation()

    # Hertraining van ElasticNet model met de uitgebreide trainingsdata inclusief 't+2'
    fac_train_extended_2 = model.factors.T
    data_train_extended_2 = extended_train_data_2_std.T

    # Debug: print output voor debugging
    print("Training extended model for t+3 with data and factors...")

    # Fit ElasticNet model met de uitgebreide trainingsdata inclusief 't+2'
    model.enet_fit(data_train_extended_2, fac_train_extended_2)

    # Controleer of het model correct is ingesteld
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")

    # Gebruik alleen de factoren van 't+2' voor het voorspellen van 't+3'
    next_timestamp_3 = next_timestamp_2 + 1
    next_timestamp_3_str = next_timestamp_3.strftime('%Y-%m')
    factor_forecast_3 = model.factor_forecast(next_timestamp_3_str, scenarios=1)

    # Zorg ervoor dat de vorm van factor_forecast_3 overeenkomt met de verwachte inputdimensie
    if factor_forecast_3.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_3.shape[1]} features")

    # Voeg de voorspelde factoren voor t+3 toe aan de matrix in de dictionary
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_3.T))

    # Voorspel de originele variabelen op basis van de voorspelde factoren van 't+2'
    predicted_variables_t3 = model.enet_predict(factor_forecast_3.reshape(1, -1))

    # Voeg de voorspelde variabelen voor t+3 toe aan de matrix in de dictionary
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t3.T))

    # print(f"Predicted variables for {next_timestamp_3_str}:\n", predicted_variables_t3)

    # Voeg de voorspelde waarden voor 't+3' toe aan de trainingsdata
    extended_train_data_3 = np.hstack((extended_train_data, predicted_variables_t3.T))

    # Standaardiseer opnieuw de uitgebreide dataset
    extended_train_data_3_std = standardize(extended_train_data_3.T).T

    # Update de index met een nieuwe tijdstempel
    extended_index_3 = extended_index + [next_timestamp_3]

    # Zet de uitgebreide dataset om naar een pandas DataFrame met een correcte index
    extended_train_df_3 = pd.DataFrame(extended_train_data_3_std, index=Y_train.index, columns=extended_index_3)

    # Fit het model opnieuw met de uitgebreide trainingsset inclusief 't+3' voorspellingen
    model = DynamicFactorModel(extended_train_df_3, num_factors)
    model.std_data = extended_train_data_3_std.T
    model.apply_pca()
    model.yw_estimation()

    # Hertraining van ElasticNet model met de uitgebreide trainingsdata inclusief 't+3'
    fac_train_extended_3 = model.factors.T
    data_train_extended_3 = extended_train_data_3_std.T

    # Debug: print output voor debugging
    print("Training extended model for t+4 with data and factors...")

    # Fit ElasticNet model met de uitgebreide trainingsdata inclusief 't+3'
    model.enet_fit(data_train_extended_3, fac_train_extended_3)

    # Controleer of het model correct is ingesteld
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")

    # Gebruik alleen de factoren van 't+3' voor het voorspellen van 't+4'
    next_timestamp_4 = next_timestamp_3 + 1
    next_timestamp_4_str = next_timestamp_4.strftime('%Y-%m')
    factor_forecast_4 = model.factor_forecast(next_timestamp_4_str, scenarios=1)

    # Zorg ervoor dat de vorm van factor_forecast_4 overeenkomt met de verwachte inputdimensie
    if factor_forecast_4.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_4.shape[1]} features")

    # Voeg de voorspelde factoren voor t+4 toe aan de matrix in de dictionary
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_4.T))

    # Voorspel de originele variabelen op basis van de voorspelde factoren van 't+4'
    predicted_variables_t4 = model.enet_predict(factor_forecast_4.reshape(1, -1))

    # Voeg de voorspelde variabelen voor t+4 toe aan de matrix in de dictionary
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t4.T))

    # print(f"Predicted variables for {next_timestamp_4_str}:\n", predicted_variables_t4)
    
	# Voeg de voorspelde waarden voor 't+4' toe aan de trainingsdata
    extended_train_data_4 = np.hstack((extended_train_data, predicted_variables_t4.T))
    extended_train_data_4_std = standardize(extended_train_data_4.T).T
    extended_index_4 = extended_index + [next_timestamp_4]
    extended_train_df_4 = pd.DataFrame(extended_train_data_4_std, index=Y_train.index, columns=extended_index_4)
    model = DynamicFactorModel(extended_train_df_4, num_factors)
    model.std_data = extended_train_data_4_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_4 = model.factors.T
    data_train_extended_4 = extended_train_data_4_std.T
    print("Training extended model for t+5 with data and factors...")
    model.enet_fit(data_train_extended_4, fac_train_extended_4)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_5 = next_timestamp_4 + 1
    next_timestamp_5_str = next_timestamp_5.strftime('%Y-%m')
    factor_forecast_5 = model.factor_forecast(next_timestamp_5_str, scenarios=1)
    if factor_forecast_5.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_5.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_5.T))
    predicted_variables_t5 = model.enet_predict(factor_forecast_5.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t5.T))
    # print(f"Predicted variables for {next_timestamp_5_str}:\n", predicted_variables_t5)  
    
	# Voeg de voorspelde waarden voor 't+5' toe aan de trainingsdata
    extended_train_data_5 = np.hstack((extended_train_data, predicted_variables_t5.T))
    extended_train_data_5_std = standardize(extended_train_data_5.T).T
    extended_index_5 = extended_index + [next_timestamp_5]
    extended_train_df_5 = pd.DataFrame(extended_train_data_5_std, index=Y_train.index, columns=extended_index_5)
    model = DynamicFactorModel(extended_train_df_5, num_factors)
    model.std_data = extended_train_data_5_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_5 = model.factors.T
    data_train_extended_5 = extended_train_data_5_std.T
    print("Training extended model for t+6 with data and factors...")
    model.enet_fit(data_train_extended_5, fac_train_extended_5)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_6 = next_timestamp_5 + 1
    next_timestamp_6_str = next_timestamp_6.strftime('%Y-%m')
    factor_forecast_6 = model.factor_forecast(next_timestamp_6_str, scenarios=1)
    if factor_forecast_6.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_6.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_6.T))
    predicted_variables_t6 = model.enet_predict(factor_forecast_6.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t6.T))
    # print(f"Predicted variables for {next_timestamp_6_str}:\n", predicted_variables_t6)  

	# Voeg de voorspelde waarden voor 't+6' toe aan de trainingsdata
    extended_train_data_6 = np.hstack((extended_train_data, predicted_variables_t6.T))
    extended_train_data_6_std = standardize(extended_train_data_6.T).T
    extended_index_6 = extended_index + [next_timestamp_6]
    extended_train_df_6 = pd.DataFrame(extended_train_data_6_std, index=Y_train.index, columns=extended_index_6)
    model = DynamicFactorModel(extended_train_df_6, num_factors)
    model.std_data = extended_train_data_6_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_6 = model.factors.T
    data_train_extended_6 = extended_train_data_6_std.T
    print("Training extended model for t+7 with data and factors...")
    model.enet_fit(data_train_extended_6, fac_train_extended_6)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_7 = next_timestamp_6 + 1
    next_timestamp_7_str = next_timestamp_7.strftime('%Y-%m')
    factor_forecast_7 = model.factor_forecast(next_timestamp_7_str, scenarios=1)
    if factor_forecast_7.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_7.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_7.T))
    predicted_variables_t7 = model.enet_predict(factor_forecast_7.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t7.T))
    # print(f"Predicted variables for {next_timestamp_7_str}:\n", predicted_variables_t7)    

	# Voeg de voorspelde waarden voor 't+7' toe aan de trainingsdata
    extended_train_data_7 = np.hstack((extended_train_data, predicted_variables_t7.T))
    extended_train_data_7_std = standardize(extended_train_data_7.T).T
    extended_index_7 = extended_index + [next_timestamp_7]
    extended_train_df_7 = pd.DataFrame(extended_train_data_7_std, index=Y_train.index, columns=extended_index_7)
    model = DynamicFactorModel(extended_train_df_7, num_factors)
    model.std_data = extended_train_data_7_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_7 = model.factors.T
    data_train_extended_7 = extended_train_data_7_std.T
    print("Training extended model for t+8 with data and factors...")
    model.enet_fit(data_train_extended_7, fac_train_extended_7)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_8 = next_timestamp_7 + 1
    next_timestamp_8_str = next_timestamp_8.strftime('%Y-%m')
    factor_forecast_8 = model.factor_forecast(next_timestamp_8_str, scenarios=1)
    if factor_forecast_8.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_8.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_8.T))
    predicted_variables_t8 = model.enet_predict(factor_forecast_8.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t8.T))
    # print(f"Predicted variables for {next_timestamp_8_str}:\n", predicted_variables_t8)

	# Voeg de voorspelde waarden voor 't+8' toe aan de trainingsdata
    extended_train_data_8 = np.hstack((extended_train_data, predicted_variables_t8.T))
    extended_train_data_8_std = standardize(extended_train_data_8.T).T
    extended_index_8 = extended_index + [next_timestamp_8]
    extended_train_df_8 = pd.DataFrame(extended_train_data_8_std, index=Y_train.index, columns=extended_index_8)
    model = DynamicFactorModel(extended_train_df_8, num_factors)
    model.std_data = extended_train_data_8_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_8 = model.factors.T
    data_train_extended_8 = extended_train_data_8_std.T
    print("Training extended model for t+9 with data and factors...")
    model.enet_fit(data_train_extended_8, fac_train_extended_8)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_9 = next_timestamp_8 + 1
    next_timestamp_9_str = next_timestamp_9.strftime('%Y-%m')
    factor_forecast_9 = model.factor_forecast(next_timestamp_9_str, scenarios=1)
    if factor_forecast_9.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_9.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_9.T))
    predicted_variables_t9 = model.enet_predict(factor_forecast_9.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t9.T))
    # print(f"Predicted variables for {next_timestamp_9_str}:\n", predicted_variables_t9)

	# Voeg de voorspelde waarden voor 't+9' toe aan de trainingsdata
    extended_train_data_9 = np.hstack((extended_train_data, predicted_variables_t9.T))
    extended_train_data_9_std = standardize(extended_train_data_9.T).T
    extended_index_9 = extended_index + [next_timestamp_9]
    extended_train_df_9 = pd.DataFrame(extended_train_data_9_std, index=Y_train.index, columns=extended_index_9)
    model = DynamicFactorModel(extended_train_df_9, num_factors)
    model.std_data = extended_train_data_9_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_9 = model.factors.T
    data_train_extended_9 = extended_train_data_9_std.T
    print("Training extended model for t+10 with data and factors...")
    model.enet_fit(data_train_extended_9, fac_train_extended_9)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_10 = next_timestamp_9 + 1
    next_timestamp_10_str = next_timestamp_10.strftime('%Y-%m')
    factor_forecast_10 = model.factor_forecast(next_timestamp_10_str, scenarios=1)
    if factor_forecast_10.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_10.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_10.T))
    predicted_variables_t10 = model.enet_predict(factor_forecast_10.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t10.T))
    # print(f"Predicted variables for {next_timestamp_10_str}:\n", predicted_variables_t10)

	# Voeg de voorspelde waarden voor 't+10' toe aan de trainingsdata
    extended_train_data_10 = np.hstack((extended_train_data, predicted_variables_t10.T))
    extended_train_data_10_std = standardize(extended_train_data_10.T).T
    extended_index_10 = extended_index + [next_timestamp_10]
    extended_train_df_10 = pd.DataFrame(extended_train_data_10_std, index=Y_train.index, columns=extended_index_10)
    model = DynamicFactorModel(extended_train_df_10, num_factors)
    model.std_data = extended_train_data_10_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_10 = model.factors.T
    data_train_extended_10 = extended_train_data_10_std.T
    print("Training extended model for t+11 with data and factors...")
    model.enet_fit(data_train_extended_10, fac_train_extended_10)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_11 = next_timestamp_10 + 1
    next_timestamp_11_str = next_timestamp_11.strftime('%Y-%m')
    factor_forecast_11 = model.factor_forecast(next_timestamp_11_str, scenarios=1)
    if factor_forecast_11.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_11.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_11.T))
    predicted_variables_t11 = model.enet_predict(factor_forecast_11.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t11.T))
    # print(f"Predicted variables for {next_timestamp_11_str}:\n", predicted_variables_t11)

	# Voeg de voorspelde waarden voor 't+11' toe aan de trainingsdata
    extended_train_data_11 = np.hstack((extended_train_data, predicted_variables_t11.T))
    extended_train_data_11_std = standardize(extended_train_data_11.T).T
    extended_index_11 = extended_index + [next_timestamp_11]
    extended_train_df_11 = pd.DataFrame(extended_train_data_11_std, index=Y_train.index, columns=extended_index_11)
    model = DynamicFactorModel(extended_train_df_11, num_factors)
    model.std_data = extended_train_data_11_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_11 = model.factors.T
    data_train_extended_11 = extended_train_data_11_std.T
    print("Training extended model for t+12 with data and factors...")
    model.enet_fit(data_train_extended_11, fac_train_extended_11)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_12 = next_timestamp_11 + 1
    next_timestamp_12_str = next_timestamp_12.strftime('%Y-%m')
    factor_forecast_12 = model.factor_forecast(next_timestamp_12_str, scenarios=1)
    if factor_forecast_12.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_12.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_12.T))
    predicted_variables_t12 = model.enet_predict(factor_forecast_12.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t12.T))
    # print(f"Predicted variables for {next_timestamp_12_str}:\n", predicted_variables_t12)

	# Voeg de voorspelde waarden voor 't+12' toe aan de trainingsdata
    extended_train_data_12 = np.hstack((extended_train_data, predicted_variables_t12.T))
    extended_train_data_12_std = standardize(extended_train_data_12.T).T
    extended_index_12 = extended_index + [next_timestamp_12]
    extended_train_df_12 = pd.DataFrame(extended_train_data_12_std, index=Y_train.index, columns=extended_index_12)
    model = DynamicFactorModel(extended_train_df_12, num_factors)
    model.std_data = extended_train_data_12_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_12 = model.factors.T
    data_train_extended_12 = extended_train_data_12_std.T
    print("Training extended model for t+13 with data and factors...")
    model.enet_fit(data_train_extended_12, fac_train_extended_12)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_13 = next_timestamp_12 + 1
    next_timestamp_13_str = next_timestamp_13.strftime('%Y-%m')
    factor_forecast_13 = model.factor_forecast(next_timestamp_13_str, scenarios=1)
    if factor_forecast_13.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_13.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_13.T))
    predicted_variables_t13 = model.enet_predict(factor_forecast_13.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t13.T))
    # print(f"Predicted variables for {next_timestamp_13_str}:\n", predicted_variables_t13)

	# Voeg de voorspelde waarden voor 't+13' toe aan de trainingsdata
    extended_train_data_13 = np.hstack((extended_train_data, predicted_variables_t13.T))
    extended_train_data_13_std = standardize(extended_train_data_13.T).T
    extended_index_13 = extended_index + [next_timestamp_13]
    extended_train_df_13 = pd.DataFrame(extended_train_data_13_std, index=Y_train.index, columns=extended_index_13)
    model = DynamicFactorModel(extended_train_df_13, num_factors)
    model.std_data = extended_train_data_13_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_13 = model.factors.T
    data_train_extended_13 = extended_train_data_13_std.T
    print("Training extended model for t+14 with data and factors...")
    model.enet_fit(data_train_extended_13, fac_train_extended_13)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_14 = next_timestamp_13 + 1
    next_timestamp_14_str = next_timestamp_14.strftime('%Y-%m')
    factor_forecast_14 = model.factor_forecast(next_timestamp_14_str, scenarios=1)
    if factor_forecast_14.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_14.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_14.T))
    predicted_variables_t14 = model.enet_predict(factor_forecast_14.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t14.T))
    # print(f"Predicted variables for {next_timestamp_14_str}:\n", predicted_variables_t14)

	# Voeg de voorspelde waarden voor 't+14' toe aan de trainingsdata
    extended_train_data_14 = np.hstack((extended_train_data, predicted_variables_t14.T))
    extended_train_data_14_std = standardize(extended_train_data_14.T).T
    extended_index_14 = extended_index + [next_timestamp_14]
    extended_train_df_14 = pd.DataFrame(extended_train_data_14_std, index=Y_train.index, columns=extended_index_14)
    model = DynamicFactorModel(extended_train_df_14, num_factors)
    model.std_data = extended_train_data_14_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_14 = model.factors.T
    data_train_extended_14 = extended_train_data_14_std.T
    print("Training extended model for t+15 with data and factors...")
    model.enet_fit(data_train_extended_14, fac_train_extended_14)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_15 = next_timestamp_14 + 1
    next_timestamp_15_str = next_timestamp_15.strftime('%Y-%m')
    factor_forecast_15 = model.factor_forecast(next_timestamp_15_str, scenarios=1)
    if factor_forecast_15.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_15.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_15.T))
    predicted_variables_t15 = model.enet_predict(factor_forecast_15.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t15.T))
    # print(f"Predicted variables for {next_timestamp_15_str}:\n", predicted_variables_t15)
    
    # Voeg de voorspelde waarden voor 't+15' toe aan de trainingsdata
    extended_train_data_15 = np.hstack((extended_train_data, predicted_variables_t15.T))
    extended_train_data_15_std = standardize(extended_train_data_15.T).T
    extended_index_15 = extended_index + [next_timestamp_15]
    extended_train_df_15 = pd.DataFrame(extended_train_data_15_std, index=Y_train.index, columns=extended_index_15)
    model = DynamicFactorModel(extended_train_df_15, num_factors)
    model.std_data = extended_train_data_15_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_15 = model.factors.T
    data_train_extended_15 = extended_train_data_15_std.T
    print("Training extended model for t+16 with data and factors...")
    model.enet_fit(data_train_extended_15, fac_train_extended_15)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_16 = next_timestamp_15 + 1
    next_timestamp_16_str = next_timestamp_16.strftime('%Y-%m')
    factor_forecast_16 = model.factor_forecast(next_timestamp_16_str, scenarios=1)
    if factor_forecast_16.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_16.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_16.T))
    predicted_variables_t16 = model.enet_predict(factor_forecast_16.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t16.T))
    # print(f"Predicted variables for {next_timestamp_16_str}:\n", predicted_variables_t16)

	# Voeg de voorspelde waarden voor 't+16' toe aan de trainingsdata
    extended_train_data_16 = np.hstack((extended_train_data, predicted_variables_t16.T))
    extended_train_data_16_std = standardize(extended_train_data_16.T).T
    extended_index_16 = extended_index + [next_timestamp_16]
    extended_train_df_16 = pd.DataFrame(extended_train_data_16_std, index=Y_train.index, columns=extended_index_16)
    model = DynamicFactorModel(extended_train_df_16, num_factors)
    model.std_data = extended_train_data_16_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_16 = model.factors.T
    data_train_extended_16 = extended_train_data_16_std.T
    print("Training extended model for t+17 with data and factors...")
    model.enet_fit(data_train_extended_16, fac_train_extended_16)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_17 = next_timestamp_16 + 1
    next_timestamp_17_str = next_timestamp_17.strftime('%Y-%m')
    factor_forecast_17 = model.factor_forecast(next_timestamp_17_str, scenarios=1)
    if factor_forecast_17.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_17.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_17.T))
    predicted_variables_t17 = model.enet_predict(factor_forecast_17.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t17.T))
    # print(f"Predicted variables for {next_timestamp_17_str}:\n", predicted_variables_t17)

	# Voeg de voorspelde waarden voor 't+17' toe aan de trainingsdata
    extended_train_data_17 = np.hstack((extended_train_data, predicted_variables_t17.T))
    extended_train_data_17_std = standardize(extended_train_data_17.T).T
    extended_index_17 = extended_index + [next_timestamp_17]
    extended_train_df_17 = pd.DataFrame(extended_train_data_17_std, index=Y_train.index, columns=extended_index_17)
    model = DynamicFactorModel(extended_train_df_17, num_factors)
    model.std_data = extended_train_data_17_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_17 = model.factors.T
    data_train_extended_17 = extended_train_data_17_std.T
    print("Training extended model for t+18 with data and factors...")
    model.enet_fit(data_train_extended_17, fac_train_extended_17)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_18 = next_timestamp_17 + 1
    next_timestamp_18_str = next_timestamp_18.strftime('%Y-%m')
    factor_forecast_18 = model.factor_forecast(next_timestamp_18_str, scenarios=1)
    if factor_forecast_18.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_18.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_18.T))
    predicted_variables_t18 = model.enet_predict(factor_forecast_18.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t18.T))
    # print(f"Predicted variables for {next_timestamp_18_str}:\n", predicted_variables_t18)

	# Voeg de voorspelde waarden voor 't+18' toe aan de trainingsdata
    extended_train_data_18 = np.hstack((extended_train_data, predicted_variables_t18.T))
    extended_train_data_18_std = standardize(extended_train_data_18.T).T
    extended_index_18 = extended_index + [next_timestamp_18]
    extended_train_df_18 = pd.DataFrame(extended_train_data_18_std, index=Y_train.index, columns=extended_index_18)
    model = DynamicFactorModel(extended_train_df_18, num_factors)
    model.std_data = extended_train_data_18_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_18 = model.factors.T
    data_train_extended_18 = extended_train_data_18_std.T
    print("Training extended model for t+19 with data and factors...")
    model.enet_fit(data_train_extended_18, fac_train_extended_18)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_19 = next_timestamp_18 + 1
    next_timestamp_19_str = next_timestamp_19.strftime('%Y-%m')
    factor_forecast_19 = model.factor_forecast(next_timestamp_19_str, scenarios=1)
    if factor_forecast_19.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_19.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_19.T))
    predicted_variables_t19 = model.enet_predict(factor_forecast_19.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t19.T))
    # print(f"Predicted variables for {next_timestamp_19_str}:\n", predicted_variables_t19)

	# Voeg de voorspelde waarden voor 't+19' toe aan de trainingsdata
    extended_train_data_19 = np.hstack((extended_train_data, predicted_variables_t19.T))
    extended_train_data_19_std = standardize(extended_train_data_19.T).T
    extended_index_19 = extended_index + [next_timestamp_19]
    extended_train_df_19 = pd.DataFrame(extended_train_data_19_std, index=Y_train.index, columns=extended_index_19)
    model = DynamicFactorModel(extended_train_df_19, num_factors)
    model.std_data = extended_train_data_19_std.T
    model.apply_pca()
    model.yw_estimation()
    fac_train_extended_19 = model.factors.T
    data_train_extended_19 = extended_train_data_19_std.T
    print("Training extended model for t+20 with data and factors...")
    model.enet_fit(data_train_extended_19, fac_train_extended_19)
    if model.model_ena is None:
        raise ValueError("ElasticNet model is not set after fitting. Check enet_fit method.")
    next_timestamp_20 = next_timestamp_19 + 1
    next_timestamp_20_str = next_timestamp_20.strftime('%Y-%m')
    factor_forecast_20 = model.factor_forecast(next_timestamp_20_str, scenarios=1)
    if factor_forecast_20.shape[1] != num_factors:
        raise ValueError(f"Expected {num_factors} features, got {factor_forecast_20.shape[1]} features")
    predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict[num_factors], factor_forecast_20.T))
    predicted_variables_t20 = model.enet_predict(factor_forecast_20.reshape(1, -1))
    predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict[num_factors], predicted_variables_t20.T))
    # print(f"Predicted variables for {next_timestamp_20_str}:\n", predicted_variables_t20)

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
results_df.to_excel('results_PCAstatic_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx', index=False)
# Sla de voorspelde matrices op als Excel-bestanden voor elk aantal factoren
for num_factors, matrix in predicted_factors_dict.items():
    pd.DataFrame(matrix).to_excel(f'predicted_factors_matrix_{num_factors}.xlsx', index=False)
    
for num_factors, matrix in predicted_variables_dict.items():
    pd.DataFrame(matrix).to_excel(f'predicted_variables_matrix_{num_factors}.xlsx', index=False)
# Print feedback naar de gebruiker
print("Results saved to results_PCAstatic_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx")
print("Predicted factors and variables matrices saved to separate Excel files for each number of factors.")