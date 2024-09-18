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

# Initialiseer de lijst met de inputdataset, dit bevat de initiële Y_train
input_dataset = Y_train.copy()

# Define the number of factors to extract
num_factors = 5  # Start met 5 factoren, je kunt dit later aanpassen
num_steps = 40    # Voorspel 40 stappen vooruit

# Lijsten om de voorspelde factoren en variabelen op te slaan
all_predicted_factors = []
all_predicted_variables = []

# --- Begin de iteratieve voorspelling met een loop ---
print("\n--- Iterative factor and variable prediction process ---")

for step in range(1, num_steps + 1):
    print(f"Step {step}: Forecasting the next time step")
    
    # --- Stap 1: Standaardiseer de inputdataset ---
    mean_train = np.mean(input_dataset.values, axis=1, keepdims=True)
    std_train = np.std(input_dataset.values, axis=1, keepdims=True)
    input_std = (input_dataset.values - mean_train) / std_train
    input_std_df = pd.DataFrame(input_std, index=input_dataset.index, columns=input_dataset.columns)
    
    # --- Stap 2: Initialiseer en pas Dynamic Factor Model toe ---
    model = DynamicFactorModel(input_std_df, num_factors)
    model.apply_pca()
    model.yw_estimation()
    
    # --- Stap 3: Bereken de B-matrix voor de voorspelling van variabelen ---
    factors_train = model.factors.T  # (num_factors, aantal observaties)
    B_matrix = np.linalg.lstsq(factors_train.T, input_std.T, rcond=None)[0]  # Shape (num_factors, aantal variabelen)
    
    # --- Stap 4: Voorspel de volgende factoren en variabelen ---
    next_factors = model.factor_forecast(num_steps=1)  # Voorspel 1 tijdstap vooruit
    predicted_x = np.dot(next_factors, B_matrix) * std_train.T + mean_train.T
    
    # --- Stap 5: Voeg de voorspelde data toe aan de inputdataset ---
    # Voeg de voorspelde variabelen als nieuwe kolom toe
    new_col = pd.DataFrame(predicted_x.T, index=input_dataset.index, columns=[f"Predicted_{step}"])
    input_dataset = pd.concat([input_dataset, new_col], axis=1)

    # Sla de voorspelde factoren en variabelen op
    all_predicted_factors.append(next_factors)
    all_predicted_variables.append(predicted_x)
    
    # Herbereken Yule-Walker voor de volgende stap (gebaseerd op de nieuwe dataset)
    model.yw_estimation()

# Zet de voorspelde factoren en variabelen om naar Pandas DataFrames
all_predicted_factors = np.vstack(all_predicted_factors)
predicted_factors_df = pd.DataFrame(all_predicted_factors, columns=[f"Factor_{i+1}" for i in range(num_factors)])

all_predicted_variables = np.vstack(all_predicted_variables)
predicted_variables_df = pd.DataFrame(all_predicted_variables, columns=[f"Variable_{i+1}" for i in range(all_predicted_variables.shape[1])])

# Debug: print de volledige set voorspelde factoren en variabelen
print(f"All predicted factors:\n{predicted_factors_df}")
print(f"All predicted variables:\n{predicted_variables_df}")

# Sla de voorspelde factoren en variabelen op als Excel-bestand voor verder gebruik
factor_output_path = os.path.join(save_directory, 'predicted_factors_lange_loop.xlsx')
predicted_factors_df.to_excel(factor_output_path, index=False)
print(f"Predicted factors saved to: {factor_output_path}")

variables_output_path = os.path.join(save_directory, 'predicted_variables_lange_loop.xlsx')
predicted_variables_df.to_excel(variables_output_path, index=False)
print(f"Predicted variables saved to: {variables_output_path}")

# Exporteer de uiteindelijke inputdataset (inclusief de voorspelde stappen)
train_output_path = os.path.join(save_directory, 'Y_train_expanded_lange_loop.xlsx')
input_dataset.to_excel(train_output_path)
print(f"Expanded Y_train saved to: {train_output_path}")
