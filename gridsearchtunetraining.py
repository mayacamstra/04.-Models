import os
import pandas as pd
import numpy as np
from data_loader import load_data, filter_data
from factor_model_try import DynamicFactorModel
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.model_selection import ParameterGrid

# Zorg ervoor dat de directory bestaat waar je de resultaten wil opslaan
save_directory = r"C:\Thesis\04. Models\PCAstatic"
os.makedirs(save_directory, exist_ok=True)

# Load and filter data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'
df_data = load_data(file_path)
filtered_df = filter_data(df_data)

# Train en validation set definiëren
Y_train = filtered_df.loc[:, :pd.Period('2019-12', freq='M')]
Y_validate = filtered_df.loc[:, pd.Period('2020-01', freq='M'):]

# Definieer parameter grid voor hyperparameteroptimalisatie
param_grid = {
    'num_factors': [2,3,5],  # Aantal factoren
    'alpha': [0.001, 0.01, 0.1, 1.0],  # Alpha van ElasticNet
    'l1_ratio': [0.1, 0.5, 0.9]  # L1-ratio van ElasticNet
}

# Eigen R²-functie
def custom_r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1 - (ss_res / ss_tot)

# Functie om de B-matrix te berekenen met ElasticNet
def fit_elastic_net(data, factors, alpha, l1_ratio):
    enet = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    enet.fit(factors, data)
    return enet.coef_, enet.intercept_

# Lijsten om resultaten op te slaan
best_params = None
best_r2 = -np.inf  # Initieer met een negatieve waarde

# Grid Search over alle hyperparameters
for params in ParameterGrid(param_grid):
    num_factors = params['num_factors']
    alpha = params['alpha']
    l1_ratio = params['l1_ratio']

    print(f"Testing parameters: num_factors={num_factors}, alpha={alpha}, l1_ratio={l1_ratio}")
    
    r2_in_sample_scores = []
    r2_out_sample_scores = []

    # 5 splits toepassen
    for split in range(5):
        if split == 0:
            Y_train_split = Y_train.iloc[:, :240]
            Y_test_split = Y_train.iloc[:, 240:300]
        elif split == 1:
            Y_train_split = pd.concat([Y_train.iloc[:, :180], Y_train.iloc[:, 240:300]], axis=1)
            Y_test_split = Y_train.iloc[:, 180:240]
        elif split == 2:
            Y_train_split = pd.concat([Y_train.iloc[:, :120], Y_train.iloc[:, 180:300]], axis=1)
            Y_test_split = Y_train.iloc[:, 120:180]
        elif split == 3:
            Y_train_split = pd.concat([Y_train.iloc[:, :60], Y_train.iloc[:, 120:300]], axis=1)
            Y_test_split = Y_train.iloc[:, 60:120]
        elif split == 4:
            Y_train_split = Y_train.iloc[:, 60:300]
            Y_test_split = Y_train.iloc[:, :60]

        # Standaardiseer de inputdataset
        mean_train = np.mean(Y_train_split.values, axis=1, keepdims=True)
        std_train = np.std(Y_train_split.values, axis=1, keepdims=True)
        input_std = (Y_train_split.values - mean_train) / std_train
        input_std_df = pd.DataFrame(input_std, index=Y_train_split.index, columns=Y_train_split.columns)

        # Dynamic Factor Model toepassen
        model = DynamicFactorModel(input_std_df, num_factors)
        model.apply_pca()
        model.yw_estimation()

        # B-matrix berekenen met ElasticNet
        factors_train = model.factors.T
        B_matrix, intercept = fit_elastic_net(input_std.T, factors_train.T, alpha, l1_ratio)

        # Dynamic Factor Model toepassen op de test set
        mean_test = mean_train  # Gebruik de gemiddelde en std van de train set
        std_test = std_train
        Y_test_std = (Y_test_split.values - mean_test) / std_test

        # Voorspelling maken
        factors_test = model.factors[:Y_test_std.shape[1]].T
        Y_test_predicted = np.dot(factors_test.T, B_matrix.T) * std_test.T + mean_test.T

        # Zorg ervoor dat Y_test_std dezelfde vorm heeft als Y_test_predicted
        Y_test_std = Y_test_std.T  # Transponeer Y_test_std zodat het (60, 66) wordt
        
        # R² berekenen
        r2_test = custom_r2_score(Y_test_std, Y_test_predicted)
        r2_out_sample_scores.append(r2_test)

    # Gemiddelde R² over alle splits
    avg_r2_out_sample = np.mean(r2_out_sample_scores)
    print(f"Average out-of-sample R² for current parameters: {avg_r2_out_sample}")

    # Update de beste parameters als de huidige setup beter is
    if avg_r2_out_sample > best_r2:
        best_r2 = avg_r2_out_sample
        best_params = params

# Toon de beste hyperparameters en bijbehorende R²
print(f"\nBest hyperparameters: {best_params}")
print(f"Best average out-of-sample R²: {best_r2}")
