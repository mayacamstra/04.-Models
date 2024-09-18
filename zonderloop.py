import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Zorg ervoor dat de directory bestaat waar we de resultaten gaan opslaan
save_directory = r"C:\Thesis\04. Models\PCAstatic"  # Map waar je de Excel-bestanden wil opslaan
os.makedirs(save_directory, exist_ok=True)

# Zorg ervoor dat de directory bestaat waar we de plots gaan opslaan
plot_dir = os.path.join(save_directory, "plots_PCAstatic")
os.makedirs(plot_dir, exist_ok=True)

# Load and filter data
file_path = 'C:/Thesis/03. Data/Final version data/Static.xlsx'

# Functie om data te laden, stel voor dat dit werkt zoals je eerdere code
def load_data(file_path):
    return pd.read_excel(file_path, index_col=0)

# Functie om data te filteren, stel voor dat deze werkt zoals in je eerdere code
def filter_data(df):
    # Placeholder filterfunctie
    return df

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

# ---- Bereken de factoren via PCA ----
from sklearn.decomposition import PCA

num_factors = 5  # We willen 5 factoren extraheren
pca = PCA(n_components=num_factors)
factors = pca.fit_transform(Y_train_std.T)  # PCA toegepast op de transpositie van de data

# Debug: Print de vorm van de geëxtraheerde factoren
print(f"Shape of extracted factors from PCA: {factors.shape}")  # (300, 5)

# --- Voer Yule-Walker schatting uit ---
from statsmodels.tsa.vector_ar.var_model import VAR

# We gebruiken een VAR-model om de autoregressieve structuur (phi-matrix) van de factoren te bepalen
var_model = VAR(factors)
var_results = var_model.fit(maxlags=1)  # We gebruiken een VAR(1) model
phi_matrix = var_results.coefs[0]  # Dit is de phi-matrix van de VAR(1)
intercept = var_results.intercept  # De intercept van het model

# Debug: Print de phi-matrix
print(f"Yule-Walker estimation (phi matrix):\n{phi_matrix}")
print(f"Intercept:\n{intercept}")

# ---- Bereken de B-matrix via lineaire regressie ----
# We gaan nu de B-matrix berekenen door lineaire regressie van de factoren op de originele variabelen.

# Gebruik gestandaardiseerde data en de factoren om de regressie te doen
B_matrix = np.zeros((Y_train_std.shape[0], num_factors))

# Voor elke variabele in Y_train_std passen we een lineaire regressie toe
for i in range(Y_train_std.shape[0]):
    # Lineaire regressie: Y_train_std[i, :] = B * factors.T
    reg = LinearRegression()
    reg.fit(factors, Y_train_std[i, :])
    B_matrix[i, :] = reg.coef_

# Debug: Print de B-matrix
print(f"Shape of B_matrix: {B_matrix.shape}")

# ---- Begin de iteratieve voorspelling voor 3 stappen vooruit ----
def predict_factors(factors_current, phi_matrix, intercept):
    return np.dot(factors_current, phi_matrix.T) + intercept

def predict_variables(factors_next, B_matrix, mean_train, std_train):
    # Bereken voorspelde variabelen op basis van f_{t+1} en B-matrix
    predicted_x = np.dot(factors_next, B_matrix.T) * std_train.T + mean_train.T
    return predicted_x

# Lijsten om de voorspelde factoren en variabelen op te slaan
all_predicted_factors = []
all_predicted_variables = []

# Stap 1: Voorspel de eerste tijdstap vooruit
print("Step 1: Forecasting the next time step")
factors_t1 = predict_factors(factors[-1, :], phi_matrix, intercept)
print(f"Predicted factors (f_{t+1}): {factors_t1}")

# Voorspel de variabelen x_{t+1}
predicted_x_t1 = predict_variables(factors_t1, B_matrix, mean_train, std_train)
print(f"Predicted variables (x_{t+1}): {predicted_x_t1}")

# Sla op in de lijsten
all_predicted_factors.append(factors_t1)
all_predicted_variables.append(predicted_x_t1)

# Stap 2: Voorspel de tweede tijdstap vooruit
print("Step 2: Forecasting the next time step")
factors_t2 = predict_factors(factors_t1, phi_matrix, intercept)
print(f"Predicted factors (f_{t+2}): {factors_t2}")

# Voorspel de variabelen x_{t+2}
predicted_x_t2 = predict_variables(factors_t2, B_matrix, mean_train, std_train)
print(f"Predicted variables (x_{t+2}): {predicted_x_t2}")

# Sla op in de lijsten
all_predicted_factors.append(factors_t2)
all_predicted_variables.append(predicted_x_t2)

# Stap 3: Voorspel de derde tijdstap vooruit
print("Step 3: Forecasting the next time step")
factors_t3 = predict_factors(factors_t2, phi_matrix, intercept)
print(f"Predicted factors (f_{t+3}): {factors_t3}")

# Voorspel de variabelen x_{t+3}
predicted_x_t3 = predict_variables(factors_t3, B_matrix, mean_train, std_train)
print(f"Predicted variables (x_{t+3}): {predicted_x_t3}")

# Sla op in de lijsten
all_predicted_factors.append(factors_t3)
all_predicted_variables.append(predicted_x_t3)

# --- Opslaan van de resultaten ---
# Zet de voorspelde factoren om naar een Pandas DataFrame voor export of verdere analyse
predicted_factors_df = pd.DataFrame(all_predicted_factors, columns=[f"Factor_{i+1}" for i in range(num_factors)])
predicted_variables_df = pd.DataFrame(np.vstack(all_predicted_variables).T, index=Y_train.index)

# Debug: print de volledige set voorspelde factoren en variabelen
print(f"All predicted factors:\n{predicted_factors_df}")
print(f"All predicted variables:\n{predicted_variables_df}")

# Sla de voorspelde factoren en variabelen op in Excel-bestanden
predicted_factors_df.to_excel(os.path.join(save_directory, 'predicted_factors.xlsx'), index=False)
predicted_variables_df.to_excel(os.path.join(save_directory, 'predicted_variables.xlsx'), index=False)

print(f"Predicted factors saved to: {os.path.join(save_directory, 'predicted_factors.xlsx')}")
print(f"Predicted variables saved to: {os.path.join(save_directory, 'predicted_variables.xlsx')}")
