import os
import pandas as pd
import numpy as np
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
model.apply_pca()  # Extract factors
model.yw_estimation()  # Estimate Phi matrix with Yule-Walker

# Initialiseer het Individual Model
ind_model = IndividualModel(Y_train_std, num_factors)

# Bereken de B-matrix op basis van de training set
B_matrix, _, _, _ = ind_model.train(Y_train_std.T, None)

# Debug: Print de vorm van de geëxtraheerde factoren
print(f"Shape of extracted factors: {model.factors.shape}")

# --- Stap 1: Voorspel f_{t+1} en x_{t+1} ---
print("\n--- Step 1: Predict f_{t+1} and x_{t+1} ---")

# Voorspel f_{t+1} met de laatste beschikbare factoren
next_factors_t1 = model.factor_forecast(num_steps=1)
print(f"Predicted factors for t+1: {next_factors_t1.shape}")

# Bereken x_{t+1} met de voorspelde factoren en B-matrix
predicted_x_t1 = np.dot(next_factors_t1, B_matrix.T) * std_train.T + mean_train.T
print(f"Predicted x_{t+1} variables: {predicted_x_t1.shape}")

# Voeg de voorspelde variabelen x_{t+1} toe aan de dataset
Y_extended_t1 = np.hstack([Y_train_std, predicted_x_t1.T])
print(f"Shape of Y_extended after adding x_{t+1}: {Y_extended_t1.shape}")

# --- Stap 2: Voorspel f_{t+2} en x_{t+2} ---
print("\n--- Step 2: Predict f_{t+2} and x_{t+2} ---")

# Standaardiseer de uitgebreide dataset met x_{t+1}
Y_extended_t1_std = (Y_extended_t1 - mean_train) / std_train

# Update het DynamicFactorModel met de nieuwe gestandaardiseerde data
model.std_data = Y_extended_t1_std  # Update model data
model.apply_pca()  # Extract factors for the new data
model.yw_estimation()  # Re-estimate Phi matrix

# Voorspel f_{t+2} met de laatste beschikbare factoren
next_factors_t2 = model.factor_forecast(num_steps=1)
print(f"Predicted factors for t+2: {next_factors_t2.shape}")

# Bereken x_{t+2} met de voorspelde factoren en B-matrix
predicted_x_t2 = np.dot(next_factors_t2, B_matrix.T) * std_train.T + mean_train.T
print(f"Predicted x_{t+2} variables: {predicted_x_t2.shape}")

# Voeg de voorspelde variabelen x_{t+2} toe aan de dataset
Y_extended_t2 = np.hstack([Y_extended_t1, predicted_x_t2.T])
print(f"Shape of Y_extended after adding x_{t+2}: {Y_extended_t2.shape}")

# --- Eindresultaat ---
# Je hebt nu de voorspelde variabelen voor t+1 en t+2 berekend en toegevoegd aan de dataset.
