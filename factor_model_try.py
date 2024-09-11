import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import r2_score
from PCA_model import apply_pca
from PLS_model import PLSModel
from utils import standardize
from datetime import datetime

class DynamicFactorModel:
    def __init__(self, df_data, num_factors, method='PCA'):
        self.df_data = df_data
        self.num_factors = num_factors
        self.method = method
        self.std_data = standardize(df_data.values.T).T
        self.factors = None
        self.phi = None
        self.B_mat = None
        self.model_ena = None

        if method == 'PCA':
            self.factor_extraction_func = self.apply_pca
        elif method == 'PLS':
            self.pls_model = PLSModel(num_factors)
            self.factor_extraction_func = self.apply_pls
        else:
            raise ValueError("Method must be 'PCA' or 'PLS'")

    def apply_pca(self):
        self.factors = apply_pca(self.std_data.T, self.num_factors)

    def apply_pls(self, X, Y):
        self.factors = self.pls_model.fit_transform(X, Y)
        if len(self.factors.shape) == 1:
            self.factors = self.factors.reshape(-1, 1)
        return self.factors

    def yw_estimation(self):
        print(f"Shape of factors before VAR: {self.factors.shape}")
        if self.factors.ndim == 1:
            raise ValueError("Factors should be a 2D array but received a 1D array.")
        elif self.factors.shape[0] != self.num_factors:
            raise ValueError(f"Expected {self.num_factors} factors, got {self.factors.shape[0]}")

        model = sm.tsa.VAR(self.factors.T)
        results = model.fit(1)
        self.phi = results.params

    def enet_fit(self, data_train, fac_train):
        self.model_ena = MultiTaskElasticNetCV(cv=5)
        self.model_ena.fit(fac_train, data_train)
        self.B_mat = self.model_ena.coef_
        x_hat = self.model_ena.predict(fac_train)
        intercept = self.model_ena.intercept_
        r2_insample = r2_score(data_train, x_hat)
        return self.B_mat, r2_insample, intercept

    def enet_predict(self, fac_predict):
        x_hat = self.model_ena.predict(fac_predict)
        return x_hat

    def factor_forecast(self, future_date, scenarios=100):
        future_date = pd.to_datetime(future_date, format='%Y-%m').to_period('M')

        current_date = self.df_data.columns[-1]

        if not isinstance(current_date, pd.Period):
            print(f"Converting current_date from {current_date} to Period format.")
            try:
                # Check if the current_date is NaT before conversion
                if pd.isna(current_date):
                    raise ValueError(f"Invalid current_date: {current_date}. Cannot convert to period.")
                current_date = pd.to_datetime(str(current_date), errors='coerce').to_period('M')

                if pd.isna(current_date):
                    raise ValueError(f"Invalid current_date detected: {self.df_data.columns[-1]}")
            except Exception as e:
                print(f"Error during current_date conversion: {e}")
                raise

        if future_date <= current_date:
            raise ValueError("The future date must be later than the last date in the data.")

        num_months = (future_date.year - current_date.year) * 12 + future_date.month - current_date.month
        if num_months != scenarios:
            print(f"Warning: Number of months ({num_months}) does not match expected scenarios ({scenarios}).")

        phi = self.phi[1:].T
        intercept = self.phi[0]
        factors_forecast = []
        factors = self.factors.T[-1]

        for i in range(num_months):
            factors = np.dot(phi, factors) + intercept
            factors_forecast.append(factors)

        return np.array(factors_forecast)
