import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime

def read_and_preprocess_data(path):
    data = pd.read_excel(path, engine='openpyxl', index_col=0)
    data.index = pd.to_datetime(data.index, dayfirst=True)  # Set dayfirst=True
    return data

def standardize_data(data_values):
    scaler = StandardScaler()
    return scaler.fit_transform(data_values)

class DynamicFactorModel:
    def __init__(self, df_data, num_factors):
        self.df_data = df_data
        self.num_factors = num_factors
        self.std_data = standardize_data(df_data.values)
        self.pca = PCA(n_components=num_factors)
        self.factors = None
        self.phi = None
        self.Lambda = None
        self.Sigma_x = None

    def apply_pca(self):
        self.factors = self.pca.fit_transform(self.std_data.T).T

    def yw_estimation(self):
        model = sm.tsa.VAR(self.factors.T)
        results = model.fit(1)
        self.phi = results.params

    def var_regression(self):
        factors = self.factors.T
        variables = self.std_data.T
        self.Lambda, _, _, _ = np.linalg.lstsq(factors, variables, rcond=None)
        residuals = variables - np.dot(factors, self.Lambda)
        self.Sigma_x = np.cov(residuals.T)

    def dfm_fit_pcayw(self):
        self.apply_pca()
        self.yw_estimation()
        self.var_regression()

    def factor_forecast(self, future_date, scenarios=100):
        future_date = datetime.strptime(future_date, '%d/%m/%Y')
        current_date = self.df_data.index[-1]
        num_months = (future_date.year - current_date.year) * 12 + future_date.month - current_date.month
        
        phi = self.phi[1:].T
        intercept = self.phi[0]
        factors_forecast = []
        factors = self.factors.T[-1]
        
        for _ in range(num_months):
            factors = np.dot(phi, factors) + intercept
            factors_forecast.append(factors)
        
        factors_forecast_df = pd.DataFrame(factors_forecast, columns=[f'Factor {i+1}' for i in range(self.num_factors)])
        
        scenarios_df = pd.concat([factors_forecast_df + np.random.multivariate_normal(np.zeros(self.num_factors), self.Sigma_x[:self.num_factors, :self.num_factors], num_months) for _ in range(scenarios)], axis=1)
        
        return factors_forecast_df, scenarios_df

    def predict_original_variables(self, factor_forecast_df):
        if self.Lambda is None:
            raise ValueError("Lambda matrix not computed. Please run var_regression first.")
        
        factor_forecast_values = factor_forecast_df.values
        
        # Debugging: print shapes
        print("Shape of factor_forecast_values:", factor_forecast_values.shape)
        print("Shape of self.Lambda:", self.Lambda.shape)
        
        # Ensure shapes are aligned correctly for matrix multiplication
        if factor_forecast_values.shape[1] != self.Lambda.shape[0]:
            raise ValueError(f"Shape mismatch: factor_forecast_values.shape[1] ({factor_forecast_values.shape[1]}) != self.Lambda.shape[0] ({self.Lambda.shape[0]})")
        
        original_var_forecast = np.dot(factor_forecast_values, self.Lambda.T)
        original_var_forecast_df = pd.DataFrame(data=original_var_forecast, columns=self.df_data.columns)
        original_var_forecast_df.index = pd.date_range(start=self.df_data.index[-1] + pd.DateOffset(months=1), periods=original_var_forecast_df.shape[0], freq='M')
        return original_var_forecast_df

def plot_original_variable_forecast(forecast_df, variable_name):
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df.index, forecast_df[variable_name], label='Forecast', color='blue')
    plt.xlabel('Time')
    plt.ylabel(variable_name)
    plt.title(f'Forecast of {variable_name}')
    plt.legend()
    plt.show()

# Main script
path_static_transposed = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Final version data\Static_T.xlsx"
data_values_static_transposed = read_and_preprocess_data(path_static_transposed)

# Apply DFM to transposed static dataset
dfm_static = DynamicFactorModel(df_data=data_values_static_transposed, num_factors=9)
dfm_static.dfm_fit_pcayw()

# Forecast
future_date = '31/12/2025'
scenarios = 100
forecast_df_static, forecast_scenarios_static = dfm_static.factor_forecast(future_date=future_date, scenarios=scenarios)
original_var_forecast_df_static = dfm_static.predict_original_variables(forecast_df_static)

print(original_var_forecast_df_static.head())

# Plot the forecast for a variable, e.g., 'CPI_Australia'
plot_original_variable_forecast(original_var_forecast_df_static, 'CPI_Australia')
