import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, filter_data
from utils import calculate_aic_bic, log_likelihood, adjusted_r2
from factor_model_try import DynamicFactorModel
from sklearn.linear_model import MultiTaskElasticNetCV

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

# Definieer het aantal factoren
num_factors = 5  # Je kunt dit aanpassen

# --- Definieer de schuivende train-test splits zoals je hebt beschreven ---
n_splits = 5
split_size = 60  # Laatste 60 maanden telkens als test set
train_size = 240  # Altijd 240 maanden trainen

# Lijsten om R² resultaten op te slaan
r2_in_sample_scores = []
r2_out_sample_scores = []

# ElasticNet functie om de B-matrix te schatten
def fit_elastic_net(data, factors):
    enet = MultiTaskElasticNetCV(cv=5, random_state=0)
    enet.fit(factors, data)
    B_matrix = enet.coef_
    intercept = enet.intercept_
    r2_in_sample = enet.score(factors, data)  # R² score to evaluate the fit
    return B_matrix, intercept, r2_in_sample

# Eigen R²-functie
def custom_r2_score(y_true, y_pred):
    """
    Eigen R²-functie die rekening houdt met de vormen van y_true en y_pred.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1 - (ss_res / ss_tot)

# --- Begin Time Series Cross-Validation ---
for split in range(n_splits):
    print(f"\n--- Split {split + 1}: Expanding window with 240 months training ---")
    
    if split == 0:
        # Split 1: Train op maanden 1-240, test op maanden 241-300
        Y_train_split = Y_train.iloc[:, :240]  # Eerste 240 maanden als train set
        Y_test_split = Y_train.iloc[:, 240:300]  # Laatste 60 maanden als test set
    
    elif split == 1:
        # Split 2: Train op maanden 1-180 en 241-300, test op maanden 181-240
        Y_train_split = pd.concat([Y_train.iloc[:, :180], Y_train.iloc[:, 240:300]], axis=1)
        Y_test_split = Y_train.iloc[:, 180:240]
    
    elif split == 2:
        # Split 3: Train op maanden 1-120 en 181-300, test op maanden 121-180
        Y_train_split = pd.concat([Y_train.iloc[:, :120], Y_train.iloc[:, 180:300]], axis=1)
        Y_test_split = Y_train.iloc[:, 120:180]
    
    elif split == 3:
        # Split 4: Train op maanden 1-60 en 121-300, test op maanden 61-120
        Y_train_split = pd.concat([Y_train.iloc[:, :60], Y_train.iloc[:, 120:300]], axis=1)
        Y_test_split = Y_train.iloc[:, 60:120]
    
    elif split == 4:
        # Split 5: Train op maanden 61-300, test op maanden 1-60
        Y_train_split = Y_train.iloc[:, 60:300]
        Y_test_split = Y_train.iloc[:, :60]
    
    # --- Stap 1: Standaardiseer de inputdataset ---
    mean_train = np.mean(Y_train_split.values, axis=1, keepdims=True)
    std_train = np.std(Y_train_split.values, axis=1, keepdims=True)
    input_std = (Y_train_split.values - mean_train) / std_train
    input_std_df = pd.DataFrame(input_std, index=Y_train_split.index, columns=Y_train_split.columns)

    # --- Stap 2: Initialiseer en pas Dynamic Factor Model toe ---
    model = DynamicFactorModel(input_std_df, num_factors)
    model.apply_pca()
    model.yw_estimation()

    # --- Stap 3: Bereken de B-matrix met ElasticNet ---
    factors_train = model.factors.T  # (num_factors, aantal observaties)
    B_matrix, intercept, r2_in_sample = fit_elastic_net(input_std.T, factors_train.T)
    print(f"ElasticNet in-sample R²: {r2_in_sample}")
    r2_in_sample_scores.append(r2_in_sample)

    # --- Stap 4: Test de performance op de test set ---
    mean_test = mean_train  # Gebruik de mean en std van de train set
    std_test = std_train
    Y_test_std = (Y_test_split.values - mean_test) / std_test
    
    # Debugging: Check de vorm van voorspelde en echte waarden
    # De testfactoren moeten overeenkomen met de testperiode, dus selecteer de juiste factoren.
    factors_test = model.factors[:Y_test_std.shape[1]].T  # Pak de factoren voor de testperiode
    
    # Controleer de dimensies voordat we vermenigvuldigen
    print(f"Shape of factors_test: {factors_test.T.shape}")  # Verwacht (60, num_factors)
    print(f"Shape of B_matrix.T: {B_matrix.T.shape}")  # Verwacht (num_factors, 66)

    Y_test_predicted = np.dot(factors_test.T, B_matrix.T) * std_test.T + mean_test.T
    
    # Debugging: Check de vorm
    print(f"Shape of Y_test_predicted: {Y_test_predicted.shape}")  # Check of deze overeenkomt met Y_test_std
    print(f"Shape of Y_test_std: {Y_test_std.shape}")  # Test set shape
    
    # Zorg ervoor dat Y_test_std dezelfde vorm heeft als Y_test_predicted
    Y_test_std = Y_test_std.T  # Transponeer Y_test_std zodat het (60, 66) wordt
    
    # Gebruik de eigen R²-functie
    r2_test = custom_r2_score(Y_test_std, Y_test_predicted)
    print(f"R² on test set for split {split + 1}: {r2_test}")
    r2_out_sample_scores.append(r2_test)

# Bereken het gemiddelde van de in-sample en out-sample R² scores over alle splits
avg_r2_in_sample = np.mean(r2_in_sample_scores)
avg_r2_out_sample = np.mean(r2_out_sample_scores)
print(f"\nAverage in-sample R² across all splits: {avg_r2_in_sample}")
print(f"Average out-of-sample R² across all splits: {avg_r2_out_sample}")
