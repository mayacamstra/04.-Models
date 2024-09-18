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

# Initialiseer het Dynamic Factor Model
model = DynamicFactorModel(Y_train_std, num_factors)

# Pas PCA toe om de factoren te extraheren
model.apply_pca()

# Debug: Print de vorm van de geëxtraheerde factoren
print(f"Shape of extracted factors from PCA: {model.factors.shape}")

# Voer Yule-Walker (VAR) schatting uit op de factoren
model.yw_estimation()

# Debug: Print de Yule-Walker schattingen (phi-matrix)
print(f"Yule-Walker estimation (phi matrix):\n{model.phi}")

# Controleer de vorm van de phi-matrix
print(f"Shape of phi matrix: {model.phi.shape}")

# Excludeer de eerste rij (intercept) van phi om alleen met de dynamische matrix te werken
eigenvalues, _ = np.linalg.eig(model.phi[1:])  # phi[1:] verwijdert de intercept

# Print de eigenwaarden
if np.all(np.abs(eigenvalues) < 1):
    print(f"Initial eigenvalues are within the unit circle. Model is stable.")
else:
    print(f"Warning: Some initial eigenvalues are outside the unit circle. Model might be unstable.")

# --- Begin de iteratieve voorspelling voor 40 stappen vooruit ---
print("\n--- Iterative factor prediction process ---")
num_steps = 40  # Aantal tijdstappen om vooruit te voorspellen
all_predicted_factors = []

for step in range(1, num_steps + 1):
    print(f"Step {step}: Forecasting the next time step")
    
    # Voorspel de volgende tijdstap vooruit
    next_factors = model.factor_forecast(num_steps=1)
    print(f"Shape of next_factors at step {step}: {next_factors.shape}")

    # Voeg de voorspelling toe aan de bestaande factoren
    model.factors = np.vstack([model.factors, next_factors])
    print(f"Shape of model.factors after step {step}: {model.factors.shape}")

    # Bewaar de voorspelde factoren voor later gebruik
    all_predicted_factors.append(next_factors)

    # Herbereken Yule-Walker
    model.yw_estimation()
    eigenvalues_step, _ = np.linalg.eig(model.phi[1:])
    if np.all(np.abs(eigenvalues_step) < 1):
        print(f"Step {step}: Eigenvalues are within the unit circle. Model is stable.")
    else:
        print(f"Step {step}: Warning: Some eigenvalues are outside the unit circle. Model might be unstable.")

# Alle voorspelde factoren samenvoegen
all_predicted_factors = np.vstack(all_predicted_factors)

# Zet de voorspelde factoren om naar een Pandas DataFrame voor export of verdere analyse
predicted_factors_df = pd.DataFrame(all_predicted_factors, columns=[f"Factor_{i+1}" for i in range(num_factors)])

# Debug: print de volledige set voorspelde factoren
print(f"All predicted factors (for {num_steps} steps):\n{predicted_factors_df}")

# Sla de voorspelde factoren op in een Excel-bestand
output_excel_path = os.path.join(save_directory, 'factorforecasts5.xlsx')
predicted_factors_df.to_excel(output_excel_path, index=False)
print(f"Predicted factors saved to: {output_excel_path}")
