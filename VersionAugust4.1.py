import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import copy

def read_and_preprocess_data(path):
    data = pd.read_excel(path, engine='openpyxl', index_col=0)
    data.columns = pd.to_datetime(data.columns, format='%d/%m/%Y')
    return data

def standardize_data(data_values):
    scaler = StandardScaler()
    return scaler.fit_transform(data_values)

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

class DynamicFactorModel:
    def __init__(self, df_data, num_factors):
        self.df_data = df_data
        self.num_factors = num_factors
        self.std_data = standardize_data(df_data.values.T).T
        self.pca = PCA(n_components=num_factors)
        self.factors = None
        self.phi = None
        self.B_mat = None
        self.model_ena = None

    def apply_pca(self):
        self.factors = self.pca.fit_transform(self.std_data.T).T

    def yw_estimation(self):
        model = sm.tsa.VAR(self.factors.T)
        results = model.fit(1)
        self.phi = results.params

    def enet_fit(self, data_train, fac_train):
        self.model_ena = ElasticNet()
        self.model_ena.fit(fac_train, data_train)
        self.B_mat = self.model_ena.coef_
        x_hat = self.model_ena.predict(fac_train)
        intercept = self.model_ena.intercept_
        r2_insample = r2_score(data_train, x_hat)
        return self.B_mat, r2_insample, intercept

    def enet_predict(self, fac_predict):
        x_hat = self.model_ena.predict(fac_predict)
        return x_hat

    def autoregression(self, data_train_reg, fac_train, beta_const):
        X = data_train_reg.T
        Y = (self.std_data - np.dot(fac_train, self.B_mat.T) - beta_const).T
        Y = np.matrix(Y)
        X = np.matrix(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        return results.params

    def dfm_fit_pcayw(self, data_train, data_train_reg):
        self.apply_pca()
        self.yw_estimation()
        self.B_mat, r2_insample, beta_const = self.enet_fit(data_train, self.factors.T)
        C_matrix = self.autoregression(data_train_reg, self.factors.T, beta_const)
        return self.B_mat, C_matrix, r2_insample, beta_const

    def factor_forecast(self, future_date, scenarios=100):
        future_date = datetime.strptime(future_date, '%d/%m/%Y')
        current_date = self.df_data.columns[-1]
        if future_date <= current_date:
            raise ValueError("Future date must be greater than the current data's last date.")
        num_months = (future_date.year - current_date.year) * 12 + future_date.month - current_date.month
        
        phi = self.phi[1:].T
        intercept = self.phi[0]
        factors_forecast = []
        factors = self.factors.T[-1]
        
        for _ in range(num_months):
            factors = np.dot(phi, factors) + intercept
            factors_forecast.append(factors)
        
        return np.array(factors_forecast)

# Load and preprocess data
FILE_PATH = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis_OLD\03. Data\Final version data\Static.xlsx"
df_data = read_and_preprocess_data(FILE_PATH)

# Initialize model
model = DynamicFactorModel(df_data, num_factors=3)

# Split data for training and validation
DATE_VALIDATE = datetime.strptime('31/01/2010', '%d/%m/%Y')
print("DATE_VALIDATE:", DATE_VALIDATE)

# Find the index of the validation date
if DATE_VALIDATE in df_data.columns:
    date_index = df_data.columns.get_loc(DATE_VALIDATE)
else:
    raise ValueError(f"Date {DATE_VALIDATE} not found in dataframe columns")

# Use the index to slice the data
Y_train_PCA = df_data.iloc[:, :date_index]

# Prepare data for individual model training

REGRESSION_STEP = 12
Y_train_other = Y_train_PCA.iloc[REGRESSION_STEP:,:]
Y_reg_train = df_data.iloc[:, :date_index + 1 - REGRESSION_STEP]

# Standardize filtered data
Y_filtered_std = Y_train_other / Y_train_other.std()
Y_reg_filtered_std = Y_reg_train / Y_reg_train.std()

# Fit model
B_matrix, C_matrix, r2_insample, beta_const = model.dfm_fit_pcayw(Y_filtered_std, Y_reg_filtered_std)
print(f'R2 insample: {r2_insample}')

# Predict and evaluate
part_1 = pd.DataFrame(np.dot(model.factors.T, B_matrix.T), columns=Y_filtered_std.columns, index=Y_filtered_std.index)
part_2 = pd.DataFrame(np.dot(Y_reg_filtered_std.values, C_matrix), columns=Y_filtered_std.columns, index=Y_filtered_std.index)
Y_hat = (part_1 + part_2 + beta_const) * Y_train_other.std()

RMSE_insample = RMSE(Y_train_other, Y_hat)
R2_insample = r2_score(Y_train_other, Y_hat)
print(f'R2 insample: {R2_insample}')
print(f'RMSE insample: {RMSE_insample}')
