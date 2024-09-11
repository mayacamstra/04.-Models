import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, filter_data
from utils import standardize, RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model_try import DynamicFactorModel

# Zorg ervoor dat de directory bestaat waar we de resultaten gaan opslaan
save_directory = r"C:\Thesis\04. Models\PCAstatic"
os.makedirs(save_directory, exist_ok=True)

plot_dir = os.path.join(save_directory, "plots_PCAstatic")
os.makedirs(plot_dir, exist_ok=True)

# Load and filter data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
df_data = load_data(file_path)  # Let op: de data wordt hier al omgezet naar maandperioden
filtered_df = filter_data(df_data)

# Standaardiseer de data vóór het opsplitsen
filtered_df_std = standardize(filtered_df)

# Definieer de periode voor training en validatie
DATE_TRAIN_END = pd.Period('2019-12', freq='M')
DATE_VALIDATE_START = pd.Period('2020-01', freq='M')
DATE_VALIDATE_END = pd.Period('2023-11', freq='M')

# Split the standardized data into training and validation sets
Y_train_std = filtered_df_std.loc[:, :DATE_TRAIN_END]
Y_validate_std = filtered_df_std.loc[:, DATE_VALIDATE_START:DATE_VALIDATE_END]

factor_range = range(5, 13)
results = []
predicted_factors_dict = {}
predicted_variables_dict = {}

for num_factors in factor_range:
    print(f"\nEvaluating model with {num_factors} factors")

    # Gebruik de gestandaardiseerde trainingsdata voor PCA
    model = DynamicFactorModel(Y_train_std.T, num_factors)
    model.apply_pca()
    model.yw_estimation()

    train_split_index = int(model.factors.shape[1] * 0.8)
    data_train = Y_train_std.iloc[:, :train_split_index].T
    fac_train = model.factors[:, :train_split_index].T
    data_test = Y_train_std.iloc[:, train_split_index:].T
    fac_test = model.factors[:, train_split_index:].T

    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)
    y_hat_train = model.enet_predict(fac_train)
    y_hat_test = model.enet_predict(fac_test)

    residuals_train = data_train - y_hat_train
    residuals_test = data_test - y_hat_test
    residuals_mean = np.mean(residuals_train, axis=0)

    print(f"Mean of residuals for {num_factors} factors: {residuals_mean}")
    if np.allclose(residuals_mean, 0, atol=1e-5):
        print(f"No shift detected in the factors mu(varepsilon) = 0) for {num_factors} factors.")
    else:
        print(f"Warning: Shift detected in the factors mu(varepsilon) != 0) for {num_factors} factors.")

    current_train_data = Y_train_std
    current_index = pd.PeriodIndex(Y_train_std.columns, freq='M')

    for t in range(1, 3):
        next_timestamp = current_index[-1] + 1
        next_timestamp_str = next_timestamp.strftime('%Y-%m')
        print(f"Next timestamp calculated: {next_timestamp_str}")
        
        # Gebruik de juiste tijdstempels bij het voorspellen
        factor_forecast = model.factor_forecast(next_timestamp_str, scenarios=1)

        predicted_factors_dict[num_factors] = np.hstack(
            (predicted_factors_dict.get(num_factors, np.empty((num_factors, 0))), factor_forecast.T)
        )
        predicted_variables = model.enet_predict(factor_forecast.reshape(1, -1))
        predicted_variables_dict[num_factors] = np.hstack(
            (predicted_variables_dict.get(num_factors, np.empty((Y_train_std.shape[0], 0))), predicted_variables.T)
        )

        extended_train_data = np.hstack((current_train_data, predicted_variables.T))
        extended_train_data_std = standardize(extended_train_data.T).T
        extended_index = current_index.append(pd.PeriodIndex([next_timestamp], freq='M'))

        extended_train_df = pd.DataFrame(extended_train_data_std, index=Y_train_std.index, columns=extended_index)
        model = DynamicFactorModel(extended_train_df, num_factors)
        model.apply_pca()
        model.yw_estimation()
        model.enet_fit(extended_train_data_std.T, model.factors.T)

        current_train_data = extended_train_data_std
        current_index = extended_index

    # Berekeningen van RMSE, AIC, BIC en andere statistieken
    rmse_value_in_sample = RMSE(data_train, y_hat_train)
    rmse_value_test_sample = RMSE(data_test, y_hat_test)
    r2_test_sample = calculate_r2(data_test, y_hat_test)
    log_like_value = log_likelihood(data_train, y_hat_train)
    aic_value, bic_value = calculate_aic_bic(y_hat_train, data_train, num_factors)
    adj_r2_in_sample = adjusted_r2(r2_insample, data_train.shape[0], num_factors)
    adj_r2_test_sample = adjusted_r2(r2_test_sample, data_test.shape[0], num_factors)

    results.append({
        'Num_Factors': num_factors,
        'RMSE_InSample': rmse_value_in_sample.mean(),
        'R2_InSample': r2_insample,
        'Adjusted_R2_InSample': adj_r2_in_sample,
        'RMSE_TestSample': rmse_value_test_sample.mean(),
        'R2_TestSample': r2_test_sample,
        'Adjusted_R2_TestSample': adj_r2_test_sample,
        'Log_Likelihood': log_like_value,
        'AIC': aic_value,
        'BIC': bic_value
    })

# Opslaan van de resultaten naar een Excel-bestand
results_df = pd.DataFrame(results)
results_path = os.path.join(save_directory, 'results_PCAstatic_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx')
results_df.to_excel(results_path, index=False)

