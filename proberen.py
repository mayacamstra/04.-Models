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

predicted_df = pd.read_excel("C:/Thesis/04. Models/predicted_variables_matrix_5.xlsx")

rmse_values = RMSE(Y_validate_std, predicted_df)
r2_values = calculate_r2(Y_validate_std, predicted_df)

# Combineer de resultaten
results_df = pd.DataFrame({
    'RMSE': rmse_values,
    'R_squared': r2_values
})

# Print de resultaten
# print(results_df)

# Opslaan van resultaten in een Excel-bestand
results_df.to_excel('PCAstatic_out_of_sample_rmse_r2_factor_5.xlsx', index=False)