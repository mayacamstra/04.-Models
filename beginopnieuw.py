import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, filter_data
from utils import RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model_try import DynamicFactorModel
from individual_model_try import IndividualModel

# Zorg ervoor dat de directory bestaat waar we de resultaten gaan opslaan
save_directory = r"C:\Thesis\04. Models\PCAstatic"  # Map waar je de Excel-bestanden wil opslaan
os.makedirs(save_directory, exist_ok=True)

# Zorg ervoor dat de directory bestaat waar we de plots gaan opslaan
plot_dir = os.path.join(save_directory, "plots_PCAstatic")
os.makedirs(plot_dir, exist_ok=True)

# Load and filter data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
df_data = load_data(file_path)
print(f"Data loaded from {file_path}, shape: {df_data.shape}")

filtered_df = filter_data(df_data)
print(f"Data after filtering, shape: {filtered_df.shape}")

# Define training and validation periods and split the data
DATE_TRAIN_END = pd.Period('2019-12', freq='M')
DATE_VALIDATE_START = pd.Period('2020-01', freq='M')
DATE_VALIDATE_END = pd.Period('2023-11', freq='M')

# Split the data into training and validation sets
Y_train = filtered_df.loc[:, :DATE_TRAIN_END]  # Data until 2019-12
Y_validate = filtered_df.loc[:, DATE_VALIDATE_START:DATE_VALIDATE_END]  # Data from 2020-01 to 2023-11

# Debug: Check the shapes of the training and validation sets
print(f"Y_train shape: {Y_train.shape} (expected: 66 variables x ~300 months)")
print(f"Y_validate shape: {Y_validate.shape} (expected: 66 variables x ~47 months)")

# Bereken de mean en std van Y_train voor het standaardiseren
mean_train = np.mean(Y_train.values, axis=1, keepdims=True)
std_train = np.std(Y_train.values, axis=1, keepdims=True)

# Debug: print de gemiddelde en standaarddeviatie per rij
print(f"Mean per row (Y_train): {mean_train.flatten()}")
print(f"Standard deviation per row (Y_train): {std_train.flatten()}")

# Standaardiseer Y_train met de eigen mean en std
Y_train_std = (Y_train.values - mean_train) / std_train

# Zet Y_train_std terug naar een Pandas DataFrame met de originele index en kolommen
Y_train_std_df = pd.DataFrame(Y_train_std, index=Y_train.index, columns=Y_train.columns)

# Debug: print de eerste paar rijen van de gestandaardiseerde Y_train
print(f"First 5 rows of Y_train_std after standardization:\n{Y_train_std[:5, :5]}")

# --- Actiepunt 1: Controleer de variantie van de gestandaardiseerde data ---
variance_per_variable = np.var(Y_train_std, axis=1)
print(f"Variance per variable (Y_train_std): {variance_per_variable}")

# --- Voorspelling van factoren en variabelen ---
num_factors = 5  # Aantal te extraheren factoren
num_steps = 40  # Aantal tijdstappen om vooruit te voorspellen

# Initialiseer het Dynamic Factor Model
model = DynamicFactorModel(Y_train_std, num_factors)
model.apply_pca()
model.yw_estimation()

# Initialiseer het Individual Model
ind_model = IndividualModel(Y_train_std, num_factors)
B_matrix, _, _, _ = ind_model.train(model.factors[-1:, :], None)  # Alleen de laatste kolom (5 factoren)

# Lijsten om voorspellingen op te slaan
all_predicted_factors = []
all_predicted_variables = []

for step in range(1, num_steps + 1):
    print(f"Step {step}: Forecasting the next time step")
    
    # Voorspel de volgende tijdstap voor de factoren
    next_factors = model.factor_forecast(num_steps=1)
    print(f"Shape of next_factors at step {step}: {next_factors.shape}")

    # Voeg de voorspelling toe aan de bestaande factoren
    model.factors = np.vstack([model.factors, next_factors])
    print(f"Shape of model.factors after step {step}: {model.factors.shape}")
    
    # Sla de voorspelde factoren op
    all_predicted_factors.append(next_factors)

    # Voorspel de individuele variabelen met de B-matrix en f_{t+step}
    predicted_variables = np.dot(next_factors, B_matrix.T) * std_train.T + mean_train.T
    print(f"Shape of predicted_variables at step {step}: {predicted_variables.shape}")
    
    # Sla de voorspelde variabelen op
    all_predicted_variables.append(predicted_variables)

    # Herbereken Yule-Walker voor de volgende stap
    model.yw_estimation()
    
    # Excludeer de eerste rij (intercept) van phi om alleen met de dynamische matrix te werken
    eigenvalues, _ = np.linalg.eig(model.phi[1:])  # phi[1:] verwijdert de intercept

    # Print de eigenwaarden
    if np.all(np.abs(eigenvalues) < 1):
        print(f"Initial eigenvalues are within the unit circle. Model is stable.")
    else:
        print(f"Warning: Some initial eigenvalues are outside the unit circle. Model might be unstable.")

# Sla de voorspelde factoren op in een Excel-bestand
all_predicted_factors = np.vstack(all_predicted_factors)
predicted_factors_df = pd.DataFrame(all_predicted_factors, columns=[f"Factor_{i+1}" for i in range(num_factors)])
factor_output_path = os.path.join(save_directory, 'factorforecasts5.xlsx')
predicted_factors_df.to_excel(factor_output_path, index=False)
print(f"Predicted factors saved to: {factor_output_path}")

# Sla de voorspelde variabelen op in een Excel-bestand
all_predicted_variables = np.vstack(all_predicted_variables)
predicted_variables_df = pd.DataFrame(all_predicted_variables, columns=Y_train.index)
variables_output_path = os.path.join(save_directory, 'variables5forecast.xlsx')
predicted_variables_df.to_excel(variables_output_path, index=False)
print(f"Predicted variables saved to: {variables_output_path}")