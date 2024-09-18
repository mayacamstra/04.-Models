import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, filter_data
from utils import standardize, RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model import DynamicFactorModel

# Zorg ervoor dat de directory bestaat waar we de resultaten gaan opslaan
save_directory = r"C:\Thesis\04. Models\PCAstatic"  # Map waar je de Excel-bestanden wil opslaan
os.makedirs(save_directory, exist_ok=True)

# Zorg ervoor dat de directory bestaat waar we de plots gaan opslaan
plot_dir = os.path.join(save_directory, "plots_PCAstatic")
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

# Standardize the training set and get the mean and std used for the transformation
Y_train_std, train_mean, train_std = standardize(Y_train.values.T)

# Apply the same transformation to the validation set using the mean and std from the training set
Y_validate_std, _, _ = standardize(Y_validate.values.T, mean=train_mean, std=train_std)

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
    model.yw_estimation()
    
    # Use 80% of the training data for training and 20% for testing
    train_split_index = int(model.factors.shape[1] * 0.8)
    data_train = Y_train_std[:, :train_split_index].T
    fac_train = model.factors[:, :train_split_index].T
    data_test = Y_train_std[:, train_split_index:].T
    fac_test = model.factors[:, train_split_index:].T
    
    # Fit ElasticNet model
    B_matrix, r2_insample, intercept = model.enet_fit(data_train, fac_train)
    
    # Validate model on in-sample data
    y_hat_train = model.enet_predict(fac_train)
    y_hat_test = model.enet_predict(fac_test)
    
    # Calculate residuals
    residuals_train = data_train - y_hat_train
    residuals_test = data_test - y_hat_test
    
    # Check if the mean of the residuals (mu(varepsilon) is zero
    residuals_mean = np.mean(residuals_train, axis=0)
    print(f"Mean of residuals for {num_factors} factors: {residuals_mean}")

    if np.allclose(residuals_mean, 0, atol=1e-5):
        print(f"No shift detected in the factors mu(varepsilon) = 0) for {num_factors} factors.")
    else:
        print(f"Warning: Shift detected in the factors mu(varepsilon) != 0) for {num_factors} factors.")

    # Voorspellingen voor de validatieset zonder de trainingsset bij te werken
    for date in Y_validate.columns:
        # Voorspel de volgende set factoren voor de validatiemaand
        factor_forecast = model.factor_forecast(date.strftime('%Y-%m'), scenarios=1)
        
        if factor_forecast.shape[1] != num_factors:
            raise ValueError(f"Expected {num_factors} features, got {factor_forecast.shape[1]} features")
        
        # Voeg de voorspelde factoren toe aan de matrix in de dictionary
        predicted_factors_dict[num_factors] = np.hstack((predicted_factors_dict.get(num_factors, np.empty((num_factors, 0))), factor_forecast.T))
        
        # Voorspel de originele variabelen op basis van de voorspelde factoren
        predicted_variables = model.enet_predict(factor_forecast.reshape(1, -1))
        
        # Voeg de voorspelde variabelen toe aan de matrix in de dictionary
        predicted_variables_dict[num_factors] = np.hstack((predicted_variables_dict.get(num_factors, np.empty((Y_train_std.shape[0], 0))), predicted_variables.T))

    # Bereken RMSE en R² voor in-sample en test data
    rmse_value_in_sample = RMSE(data_train, y_hat_train)
    rmse_value_test_sample = RMSE(data_test, y_hat_test)
    r2_test_sample = calculate_r2(data_test, y_hat_test)
    
    # Calculate log-likelihood, AIC, and BIC
    log_like_value = log_likelihood(data_train, y_hat_train)
    aic_value, bic_value = calculate_aic_bic(y_hat_train, data_train, num_factors)
    
    # Calculate adjusted R²
    adj_r2_in_sample = adjusted_r2(r2_insample, data_train.shape[0], num_factors)
    adj_r2_test_sample = adjusted_r2(r2_test_sample, data_test.shape[0], num_factors)
    
    # Append the results to the list
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

    # Check eigenvalues of the Yule-Walker matrix A (phi)
    eigenvalues, _ = np.linalg.eig(model.phi[1:])  # Matrix A is phi[1:], excluding the intercept
    print(f"Eigenvalues of the matrix A (phi[1:]) for {num_factors} factors: {eigenvalues}")

    # Check if all eigenvalues have an absolute value less than 1
    if np.all(np.abs(eigenvalues) < 1):
        print(f"All eigenvalues for {num_factors} factors are within the unit circle. Model is stable.")
    else:
        print(f"Warning: Some eigenvalues for {num_factors} factors are outside the unit circle. Model might be unstable.")

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
results_path = os.path.join(save_directory, 'results_PCAstatic_with_AIC_BIC_AdjustedR2_LogLikelihood_Residuals.xlsx')
results_df.to_excel(results_path, index=False)

# Sla de voorspelde matrices op als Excel-bestanden voor elk aantal factoren
for num_factors, matrix in predicted_factors_dict.items():
    factors_path = os.path.join(save_directory, f'PCAstatic_predicted_factors_matrix_{num_factors}.xlsx')
    pd.DataFrame(matrix).to_excel(factors_path, index=False)
for num_factors, matrix in predicted_variables_dict.items():
    variables_path = os.path.join(save_directory, f'PCAstatic_predicted_variables_matrix_{num_factors}.xlsx')
    pd.DataFrame(matrix).to_excel(variables_path, index=False)

print(f"Results saved to {results_path}")
print(f"Predicted factors and variables matrices saved to separate Excel files for each number of factors.")