import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, filter_data
from utils import RMSE, calculate_r2, calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model_try import DynamicFactorModel

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

# Define the number of factors to extract
num_factors = 5  # Start met 5 factoren, je kunt dit later aanpassen
num_steps = 40    # Voorspel 40 stappen vooruit

# Initialiseer het Dynamic Factor Model
model = DynamicFactorModel(Y_train_std, num_factors)
# Pas PCA toe om de factoren te extraheren
model.apply_pca()
# Debug: Print de vorm van de geëxtraheerde factoren
print(f"Shape of extracted factors from PCA: {model.factors.shape}")

# Voer Yule-Walker (VAR) schatting uit op de factoren
model.yw_estimation()

# --- Vanaf hier gaan we expliciet de berekeningen doen voor de B-matrix en voorspellingen van variabelen ---
# Bepaal nu de B-matrix door een lineaire regressie op de gestandaardiseerde data

# De vorm van de factorenmatrix is (300, 5), transponeer om (5, 300) te krijgen
factors_train = model.factors.T  # (5, 300)

# Bereken de B-matrix met lineaire regressie: Y_train_std.T ~ factors_train
B_matrix = np.linalg.lstsq(factors_train.T, Y_train_std.T, rcond=None)[0]  # Shape (5, 66)
print(f"Shape of B_matrix: {B_matrix.shape}")

# Lijsten om de voorspelde factoren en variabelen op te slaan
all_predicted_factors = []
all_predicted_variables = []

# --- Begin de iteratieve voorspelling met een loop ---
print("\n--- Iterative factor and variable prediction process ---")

for step in range(1, num_steps + 1):
    print(f"Step {step}: Forecasting the next time step")
    
    # Voorspel de volgende factoren f_{t+step}
    next_factors = model.factor_forecast(num_steps=1)  # Voorspel 1 tijdstap vooruit
    print(f"Shape of next_factors (step {step}): {next_factors.shape}")
    
    # Bereken x_{t+step} met de voorspelde factoren en B-matrix
    predicted_x = np.dot(next_factors, B_matrix) * std_train.T + mean_train.T
    print(f"Shape of predicted x_t+{step}: {predicted_x.shape}")
    
    # Voeg de voorspelling toe aan de bestaande factoren en sla op
    model.factors = np.vstack([model.factors, next_factors])
    all_predicted_factors.append(next_factors)
    all_predicted_variables.append(predicted_x)
    
    # Herbereken Yule-Walker voor de volgende stap
    model.yw_estimation()

# Sla de voorspelde variabelen op in een Pandas DataFrame
all_predicted_factors = np.vstack(all_predicted_factors)
predicted_factors_df = pd.DataFrame(all_predicted_factors, columns=[f"Factor_{i+1}" for i in range(num_factors)])

all_predicted_variables = np.vstack(all_predicted_variables)  # Vorm (n_steps, 66)
predicted_variables_df = pd.DataFrame(all_predicted_variables, columns=[f"Variable_{i+1}" for i in range(all_predicted_variables.shape[1])])

# Debug: print de volledige set voorspelde factoren en variabelen
print(f"All predicted factors:\n{predicted_factors_df}")
print(f"All predicted variables:\n{predicted_variables_df}")

# Sla de voorspelde factoren en variabelen op als Excel-bestand voor verder gebruik
factor_output_path = os.path.join(save_directory, 'predicted_factors.xlsx')
predicted_factors_df.to_excel(factor_output_path, index=False)
print(f"Predicted factors saved to: {factor_output_path}")

variables_output_path = os.path.join(save_directory, 'predicted_variables.xlsx')
predicted_variables_df.to_excel(variables_output_path, index=False)
print(f"Predicted variables saved to: {variables_output_path}")
