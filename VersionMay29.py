# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to read and preprocess data
def read_and_preprocess_data(path):
    data = pd.read_excel(path, engine='openpyxl')
    variable_names = data.iloc[:, 0]
    data_values = data.iloc[:, 1:]
    return variable_names, data_values

# Function to standardize data
def standardize_data(data_values):
    scaler = StandardScaler()
    return scaler.fit_transform(data_values.T).T

# Function to apply Hodrick-Prescott filter
def apply_hpfilter(data):
    cycle_df = pd.DataFrame(index=data.index, columns=data.columns)
    trend_df = pd.DataFrame(index=data.index, columns=data.columns)
    for row in data.index:
        cycle, trend = sm.tsa.filters.hpfilter(data.loc[row], lamb=129600)  # lambda = 129600 for monthly data
        cycle_df.loc[row] = cycle
        trend_df.loc[row] = trend
    return cycle_df

# Function to perform PCA
def perform_pca(data, n_components=9):
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data.T).T
    return pd.DataFrame(pca_components, index=[f'PC{i+1}' for i in range(n_components)], columns=data.columns)

# Import data
path_static = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Final version data\Static.xlsx"
path_forward_looking = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Final version data\Forward.xlsx"

variable_names_static, data_values_static = read_and_preprocess_data(path_static)
variable_names_forward, data_values_forward = read_and_preprocess_data(path_forward_looking)

# Data standardizing
scaled_data_static = standardize_data(data_values_static)
scaled_data_forward = standardize_data(data_values_forward)

# Create data frames
scaled_df_static = pd.DataFrame(scaled_data_static, columns=data_values_static.columns, index=variable_names_static)
scaled_df_forward = pd.DataFrame(scaled_data_forward, columns=data_values_forward.columns, index=variable_names_forward)

# Merge into combined data frame
combined_df = pd.concat([scaled_df_static, scaled_df_forward])

# Apply Hodrick-Prescott filter
filtered_df_combined = apply_hpfilter(combined_df)
filtered_df_static = apply_hpfilter(scaled_df_static)

# Perform PCA
pca_df_combined = perform_pca(filtered_df_combined)
pca_df_static = perform_pca(filtered_df_static)

print(pca_df_combined.head())
print(pca_df_static.head())

# ------------ DIT NIET RUNNEN -----------------------------------------------------------------
# Plotting PCA components
def plot_pca_components(pca_df_combined, pca_df_static):
    plt.figure(figsize=(14, 12))

    for i in range(1, 5):
        plt.subplot(4, 1, i)
        plt.plot(pca_df_combined.columns, pca_df_combined.loc[f'PC{i}'], label='Combined Data')
        plt.plot(pca_df_static.columns, pca_df_static.loc[f'PC{i}'], label='Static Data', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel(f'Principal Component {i}')
        plt.title(f'Principal Component {i} over Time')
        plt.legend()

    for ax in plt.gcf().axes:
        ax.set_xticks(ax.get_xticks()[::12])  # Label every 12th month
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

plot_pca_components(pca_df_combined, pca_df_static)
# ------------------------ DIT NIET RUNNEN --------------------------------------------------------

# Dynamic Factor Model
class DynamicFactorModel:
    def __init__(self, df_data: pd.DataFrame, num_factors: int, timescale: int):
        self.df_data = df_data
        self.detrended_data = self.detrend_with_moving_average(self.df_data)
        self.std_data = self.standardize_df(self.df_data)
        self.timescale = timescale
        self.df_factors = None
        self.num_factors = num_factors
        self.Phi = None
        self.Sigma_f = None
        self.Lambda = None
        self.Sigma_x = None

    def detrend_with_moving_average(self, df, window=12):
        return df - df.rolling(window=window).mean()

    def standardize_df(self, df):
        return (df - df.mean()) / df.std()

    def apply_pca(self):
        pca = PCA(n_components=self.num_factors)
        principal_components = pca.fit_transform(self.std_data)
        df_factors = pd.DataFrame(index=self.std_data.index, data=principal_components, columns=[f'Factor {i+1}' for i in range(self.num_factors)])
        self.df_factors = self.standardize_df(df_factors)

    @staticmethod
    def explicit_covariance(X, Y):
        n = X.shape[0]
        return np.dot(X.T, Y) / (n - 1)

    def yw_estimation(self):
        if self.df_factors is None:
            raise ValueError("Factors not computed. Please run apply_pca first.")
        factors = self.df_factors.values
        Gamma_0 = self.explicit_covariance(factors, factors)
        Gamma_1 = self.explicit_covariance(factors[self.timescale:, :], factors[:-self.timescale, :])
        self.Phi = np.dot(Gamma_1, np.linalg.inv(Gamma_0))
        self.Sigma_f = Gamma_0 - np.dot(np.dot(self.Phi, Gamma_0), self.Phi.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.Sigma_f)
        eigenvalues[eigenvalues < 0] = 0
        self.Sigma_f = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))

    def factor_forecast(self, future_date: str, scenarios: int = 100, ci_percentiles: tuple = (2.5, 97.5)) -> pd.DataFrame:
        last_obs = self.df_factors.iloc[-1].values
        future_date = pd.to_datetime(future_date, format='%d/%m/%Y')
        forecast_length = pd.date_range(start=self.df_factors.index[-1], end=future_date, freq='YE').shape[0]
        forecast_scenarios = np.zeros((scenarios, forecast_length, self.Phi.shape[0]))
        forecast_scenarios[:, 0, :] = last_obs
        random_errors = np.random.multivariate_normal(np.zeros(self.Sigma_f.shape[0]), self.Sigma_f, (scenarios, forecast_length - 1))
        for period in range(1, forecast_length):
            forecast_scenarios[:, period, :] = np.dot(forecast_scenarios[:, period - 1, :], self.Phi.T) + random_errors[:, period - 1, :]
        mean_forecast = forecast_scenarios.mean(axis=0)
        lower_bound = np.percentile(forecast_scenarios, ci_percentiles[0], axis=0)
        upper_bound = np.percentile(forecast_scenarios, ci_percentiles[1], axis=0)
        forecast_index = pd.date_range(start=self.df_factors.index[-1], periods=forecast_length, freq='YE')
        forecast_df = pd.DataFrame(data=mean_forecast, index=forecast_index, columns=self.df_factors.columns)
        for i, column in enumerate(self.df_factors.columns):
            forecast_df[column + '_mean'] = mean_forecast[:, i]
            forecast_df[column + '_lower_bound'] = lower_bound[:, i]
            forecast_df[column + '_upper_bound'] = upper_bound[:, i]
        return forecast_df, forecast_scenarios

    def var_regression(self):
        if self.df_factors is None or self.std_data is None:
            raise ValueError("Factors or standardized data not available.")
        if self.df_factors.shape[0] != self.std_data.shape[0]:
            raise ValueError("The number of rows (time points) in df_factors and df_data must be the same.")
        factors = self.df_factors.values
        variables = self.std_data.values
        self.Lambda, _, _, _ = np.linalg.lstsq(factors, variables, rcond=None)
        residuals = variables - np.dot(factors, self.Lambda)
        self.Sigma_x = np.cov(residuals.T)

    def dfm_fit_pcayw(self):
        self.apply_pca()
        self.yw_estimation()
        self.var_regression()

    def predict_original_variables(self, factor_forecast_df):
        if self.Lambda is None:
            raise ValueError("Lambda matrix not computed. Please run var_regression first.")
        factor_forecast_values = factor_forecast_df[[col for col in factor_forecast_df.columns if '_mean' in col]].values
        original_var_forecast = np.dot(factor_forecast_values, self.Lambda.T)
        original_var_forecast_df = pd.DataFrame(data=original_var_forecast, index=factor_forecast_df.index, columns=self.df_data.columns)
        return original_var_forecast_df

# Apply DFM to static dataset
dfm_static = DynamicFactorModel(df_data=pca_df_static.T, num_factors=9, timescale=12)
dfm_static.dfm_fit_pcayw()
future_date = '31/12/2025'
scenarios = 100
forecast_df_static, forecast_scenarios_static = dfm_static.factor_forecast(future_date=future_date, scenarios=scenarios)
original_var_forecast_df_static = dfm_static.predict_original_variables(forecast_df_static)

print(original_var_forecast_df_static.head())

# Plot de forecast voor een variabele, bijvoorbeeld 'CPI_Australia'
def plot_original_variable_forecast(forecast_df, variable_name):
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df.index, forecast_df[variable_name], label='Forecast', color='blue')
    plt.xlabel('Time')
    plt.ylabel(variable_name)
    plt.title(f'Forecast of {variable_name}')
    plt.legend()
    plt.show()

plot_original_variable_forecast(original_var_forecast_df_static, 'CPI_Australia')

scaled_df_static.head()


