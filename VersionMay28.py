# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import data
path_static = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Final version data\Static.xlsx"
data_static = pd.read_excel(path_static, engine = 'openpyxl')

path_forward_looking = r"C:\Users\mayac\OneDrive - ORTEC Finance\Thesis\03. Data\Final version data\Forward.xlsx"
data_forward_looking = pd.read_excel(path_forward_looking, engine = 'openpyxl')

# Data preprocessing
variable_names_static = data_static.iloc[:, 0]
data_values_static = data_static.iloc[:, 1:]

variable_names_forward = data_forward_looking.iloc[:, 0]
data_values_forward = data_forward_looking.iloc[:, 1:]

# Data standardizing
scaler = StandardScaler()
scaled_data_static = scaler.fit_transform(data_values_static.T).T
scaled_data_forward = scaler.fit_transform(data_values_forward.T).T

# Data frame
scaled_df_static = pd.DataFrame(scaled_data_static, columns = data_values_static.columns, index = variable_names_static)
scaled_df_forward = pd.DataFrame(scaled_data_forward, columns = data_values_forward.columns, index = variable_names_forward)

# Merge into combined data frame
combined_df = pd.concat([scaled_df_static, scaled_df_forward])

# Hodrick-Prescott filter om trend eruit te halen voor de combined df
cycle_df_combined = pd.DataFrame(index=combined_df.index, columns=combined_df.columns)
trend_df_combined = pd.DataFrame(index=combined_df.index, columns=combined_df.columns)

for row in combined_df.index:
    cycle, trend = sm.tsa.filters.hpfilter(combined_df.loc[row], lamb = 129600)  # lambda = 129600 voor maandelijkse data
    cycle_df_combined.loc[row] = cycle
    trend_df_combined.loc[row] = trend

filtered_df_combined = cycle_df_combined

# PCA voor de combined df
pca_combined = PCA(n_components = 9)
pca_components_combined = pca_combined.fit_transform(filtered_df_combined.T).T
pca_df_combined = pd.DataFrame(pca_components_combined, index=[f'PC{i+1}' for i in range(9)], columns=filtered_df_combined.columns)
print(pca_df_combined.head())

# Hodrick-Prescott filter om trend eruit te halen voor de static df
cycle_df_static = pd.DataFrame(index=scaled_df_static.index, columns=scaled_df_static.columns)
trend_df_static = pd.DataFrame(index=scaled_df_static.index, columns=scaled_df_static.columns)

for row in scaled_df_static.index:
    cycle, trend = sm.tsa.filters.hpfilter(scaled_df_static.loc[row], lamb = 129600)  # lambda = 129600 voor maandelijkse data
    cycle_df_static.loc[row] = cycle
    trend_df_static.loc[row] = trend

filtered_df_static = cycle_df_static

# PCA voor static df
pca_static = PCA(n_components=9)
pca_components_static = pca_static.fit_transform(filtered_df_static.T).T
pca_df_static = pd.DataFrame(pca_components_static, index=[f'PC{i+1}' for i in range(9)], columns=filtered_df_static.columns)
print(pca_df_static.head())

# ------------------- DIT NIET RUNNEN ---------------------------------------------
# Visually highlight the differences in PC's between static and combined dataframes
plt.figure(figsize=(14, 12))

# PC1 plotten
plt.subplot(4, 1, 1)
plt.plot(pca_df_combined.columns, pca_df_combined.loc['PC1'], label='Combined Data')
plt.plot(pca_df_static.columns, pca_df_static.loc['PC1'], label='Static Data', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Principal Component 1')
plt.title('Principal Component 1 over Time')
plt.legend()

# PC2 plotten
plt.subplot(4, 1, 2)
plt.plot(pca_df_combined.columns, pca_df_combined.loc['PC2'], label='Combined Data')
plt.plot(pca_df_static.columns, pca_df_static.loc['PC2'], label='Static Data', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Principal Component 2')
plt.title('Principal Component 2 over Time')
plt.legend()

# PC3 plotten
plt.subplot(4, 1, 3)
plt.plot(pca_df_combined.columns, pca_df_combined.loc['PC3'], label='Combined Data')
plt.plot(pca_df_static.columns, pca_df_static.loc['PC3'], label='Static Data', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Principal Component 3')
plt.title('Principal Component 3 over Time')
plt.legend()

# PC4 plotten
plt.subplot(4, 1, 4)
plt.plot(pca_df_combined.columns, pca_df_combined.loc['PC4'], label='Combined Data')
plt.plot(pca_df_static.columns, pca_df_static.loc['PC4'], label='Static Data', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Principal Component 4')
plt.title('Principal Component 4 over Time')
plt.legend()

# X-as leesbaarder maken door minder frequent labels te plaatsen
for ax in plt.gcf().axes:
    ax.set_xticks(ax.get_xticks()[::12])  # Elke 12e maand labelen
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()
# --------------- DIT NIET RUNNEN ---------------------------------------------

# Dynamic Factor Model voor static df
class DynamicFactorModel:
    def __init__(self, df_data: pd.DataFrame, num_factors: int, timescale: int):
        '''
        Initialize the Dynamic Factor Model.

        Params 
        --
        df_data: pd.DataFrame containing the raw data of economic variables
        num_factors: int to specify number of factors.
        timescale: int = 1, 12, 96 corresponding to monthly, yearly, or 
        8-yearly steps in time, used in estimation and forecasting.
        '''

        # Data related initialization
        self.df_data = df_data
        self.detrended_data = self.detrend_with_moving_average(self.df_data)
        self.std_data = self.standardize_df(self.df_data)
        self.timescale = timescale

        # Factor related
        self.df_factors = None
        self.num_factors = num_factors

        # Model parameters
        self.Phi = None
        self.Sigma_f = None

    def detrend_with_moving_average(self, df, window=12):
        return df - df.rolling(window=window).mean()

    def standardize_df(self, df):
        return (df - df.mean()) / df.std()

    def apply_pca(self):
        '''
        Factor extraction from standardized data using PCA.
        '''
        pca = PCA(n_components=self.num_factors)
        principal_components = pca.fit_transform(self.std_data)
        df_factors = pd.DataFrame(index=self.std_data.index, data=principal_components, 
                            columns=[f'Factor {i+1}' for i in range(self.num_factors)])
        self.df_factors = self.standardize_df(df_factors)

    @staticmethod
    def explicit_covariance(X, Y):
        '''
        Quick helper function to compute sample covariance.
        '''
        n = X.shape[0]
        cov_matrix = np.dot(X.T, Y) / (n - 1)
        return cov_matrix

    def yw_estimation(self):
        '''
        Estimate the parameters of a VAR(1) model for the factor dynamics.
        '''
        if self.df_factors is None:
            raise ValueError("Factors not computed. Please run apply_pca first.")

        factors = self.df_factors.values

        # Computing Gamma_0 and Gamma_1
        Gamma_0 = self.explicit_covariance(factors, factors)
        Gamma_1 = self.explicit_covariance(factors[self.timescale:, :], factors[:-self.timescale, :])

        self.Phi = np.dot(Gamma_1, np.linalg.inv(Gamma_0))
        self.Sigma_f = Gamma_0 - np.dot(np.dot(self.Phi, Gamma_0), self.Phi.T)

        # Ensure Sigma_f is symmetric positive-semidefinite
        eigenvalues, eigenvectors = np.linalg.eigh(self.Sigma_f)
        eigenvalues[eigenvalues < 0] = 0
        self.Sigma_f = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))

    def factor_forecast(self, future_date: str, scenarios: int = 100, ci_percentiles: tuple = (2.5, 97.5)) -> pd.DataFrame:
        '''
        Generate forecast with confidence intervals for the VAR(1) model.
        '''
        last_obs = self.df_factors.iloc[-1].values
        future_date = pd.to_datetime(future_date, format='%d/%m/%Y')
        forecast_length = pd.date_range(start=self.df_factors.index[-1], end=future_date, freq='Y').shape[0]
        
        # Prepare matrix for all scenarios
        forecast_scenarios = np.zeros((scenarios, forecast_length, self.Phi.shape[0]))

        # First forecast is just the last observation repeated for all scenarios
        forecast_scenarios[:, 0, :] = last_obs

        # Generate random errors for all scenarios and periods at once
        random_errors = np.random.multivariate_normal(np.zeros(self.Sigma_f.shape[0]), self.Sigma_f, (scenarios, forecast_length - 1))

        # Vectorized forecast computation
        for period in range(1, forecast_length):
            forecast_scenarios[:, period, :] = np.dot(forecast_scenarios[:, period - 1, :], self.Phi.T) + random_errors[:, period - 1, :]

        # Calculate mean forecast and confidence intervals
        mean_forecast = forecast_scenarios.mean(axis=0)
        lower_bound = np.percentile(forecast_scenarios, ci_percentiles[0], axis=0)
        upper_bound = np.percentile(forecast_scenarios, ci_percentiles[1], axis=0)
        
        # Create the forecast dataframe
        forecast_index = pd.date_range(start=self.df_factors.index[-1], periods=forecast_length, freq='Y')
        forecast_df = pd.DataFrame(data=mean_forecast, index=forecast_index, columns=self.df_factors.columns)

        # Assign mean, lower, and upper bounds for each factor
        for i, column in enumerate(self.df_factors.columns):
            forecast_df[column + '_mean'] = mean_forecast[:, i]
            forecast_df[column + '_lower_bound'] = lower_bound[:, i]
            forecast_df[column + '_upper_bound'] = upper_bound[:, i]

        return forecast_df, forecast_scenarios

    def var_regression(self):
        """
        Perform regression of variables on the factors.
        """
        if self.df_factors is None or self.std_data is None:
            raise ValueError("Factors or standardized data not available.")
        
        if self.df_factors.shape[0] != self.std_data.shape[0]:
            raise ValueError("The number of rows (time points) in df_factors and df_data must be the same.")

        factors = self.df_factors.values
        variables = self.std_data.values

        # Using np.linalg.lstsq for a more numerically stable solution than np.linalg.solve
        self.Lambda, _, _, _ = np.linalg.lstsq(factors, variables, rcond=None)

        # Calculate residuals to estimate Sigma_x
        residuals = variables - np.dot(factors, self.Lambda)
        self.Sigma_x = np.cov(residuals.T)

    def dfm_fit_pcayw(self):
        '''
        Fits standardized data to DFM dynamics using 
        PCA and Yule-Walker estimation.
        '''
        self.apply_pca()
        self.yw_estimation()
        self.var_regression()

# Instantieer en pas DFM toe op statische dataset
dfm_static = DynamicFactorModel(df_data=pca_df_static.T, num_factors=9, timescale=12)

# Fit het model
dfm_static.dfm_fit_pcayw()

# Forecast tot een toekomstige datum
future_date = '31/12/2025'
scenarios = 100

forecast_df_static, forecast_scenarios_static = dfm_static.factor_forecast(future_date=future_date, scenarios=scenarios)

# Bekijk de forecast
print(forecast_df_static.head())



