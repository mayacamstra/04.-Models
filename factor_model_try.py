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
        
        # Zorg ervoor dat df_data een pandas DataFrame is
        if isinstance(df_data, np.ndarray):
            # Zet de numpy array om naar een DataFrame, gebruik makend van fictieve index en kolommen
            df_data = pd.DataFrame(df_data)       
        
        self.df_data = df_data
        self.num_factors = num_factors
        self.method = method
        self.std_data = df_data
        self.factors = None
        self.phi = None
        self.B_mat = None
        self.model_ena = None
        self.pca_model = None

        if method == 'PCA':
            self.factor_extraction_func = self.apply_pca
        elif method == 'PLS':
            self.pls_model = PLSModel(num_factors)
            self.factor_extraction_func = self.apply_pls
        else:
            raise ValueError("Method must be 'PCA' or 'PLS'")

    def apply_pca(self):
        """
        Apply Principal Component Analysis (PCA) to extract factors.
        """
        # Ontvang zowel de factoren als het PCA-model van apply_pca
        self.factors, self.pca_model = apply_pca(self.std_data.T, self.num_factors, return_model=True)


    def apply_pls(self, X, Y):
        """
        Apply PLS regression to extract factors.

        Parameters:
        X (np.ndarray): Input data for PLS.
        Y (np.ndarray): Output data for PLS (often the same as X for unsupervised factor extraction).

        Returns:
        np.ndarray: The extracted factors.
        """
        self.factors = self.pls_model.fit_transform(X, Y)
        
        # Debug: Check if factors are 2D after PLS
        print(f"Shape of PLS factors: {self.factors.shape}")

        # Ensure the factors remain 2D, if necessary reshape or transpose
        if len(self.factors.shape) == 1:
            self.factors = self.factors.reshape(-1, 1)  # Ensure it's at least 2D
        return self.factors

    def transform(self, X):
        """
        Transform the data using the fitted PLS model.

        Parameters:
        X (np.ndarray): Input data to transform using the fitted PLS model.
        
        Returns:
        np.ndarray: The transformed data.
        """
        if self.method == 'PLS':
            return self.pls_model.transform(X)
        else:
            raise ValueError("Transform method is only applicable when method='PLS'")

    def yw_estimation(self):
        """
        Perform Yule-Walker estimation on the factors to fit a VAR model.
        """
        # Debug: Check the shape of factors before transposition
        print(f"Shape of factors before VAR: {self.factors.shape}")

        # Transponeer de factorenmatrix om de juiste dimensies te krijgen (num_factors, num_time_points)
        factors_transposed = self.factors.T  # Maak de matrix (5, 300)

        # Debug: Controleer de nieuwe vorm
        print(f"Shape of factors after transposition: {factors_transposed.shape}")

        # Nu passen we het VAR-model toe op de getransponeerde factoren
        model = sm.tsa.VAR(factors_transposed)
        results = model.fit(1)
        self.phi = results.params

        # Debug: Print de geschatte phi-matrix
        print(f"Yule-Walker estimation (phi matrix):\n{self.phi}")

    def enet_fit(self, data_train, fac_train):
        """
        Fit a MultiTask ElasticNet model to the factors.
        """
        self.model_ena = MultiTaskElasticNetCV(cv=5)
        self.model_ena.fit(fac_train, data_train)
        self.B_mat = self.model_ena.coef_
        x_hat = self.model_ena.predict(fac_train)
        intercept = self.model_ena.intercept_
        r2_insample = r2_score(data_train, x_hat)
        return self.B_mat, r2_insample, intercept

    def enet_predict(self, fac_predict):
        """
        Predict using the trained MultiTask ElasticNet model.
        """
        x_hat = self.model_ena.predict(fac_predict)
        return x_hat

    def autoregression(self, data_train_reg, fac_train, beta_const):
        """
        Perform autoregression using Ordinary Least Squares (OLS).
        """
        X = data_train_reg.T
        Y = (self.std_data.T - np.dot(fac_train, self.B_mat.T) - beta_const).T

        if X.shape[1] != Y.shape[1]:
            Y = Y[:, :X.shape[1]]

        if X.shape[0] != Y.shape[0]:
            Y = Y[:X.shape[0], :]

        Y = np.matrix(Y)
        X = np.matrix(X)

        if X.shape[0] != Y.shape[0]:
            raise ValueError("The number of rows in X and Y must be equal after transposing")

        model = sm.OLS(Y, X)
        results = model.fit()
        return results.params

    def dfm_fit(self, data_train, data_train_reg=None):
        """
        Fit the Dynamic Factor Model using the selected method (PCA or PLS).
        """
        self.factor_extraction_func()  # Apply PCA or PLS based on the method
        self.yw_estimation()
        self.B_mat, r2_insample, beta_const = self.enet_fit(data_train, self.factors.T)

        if data_train_reg is not None:
            C_matrix = self.autoregression(data_train_reg, self.factors.T, beta_const)
            return self.B_mat, C_matrix, r2_insample, beta_const
        else:
            return self.B_mat, r2_insample, beta_const

    def factor_forecast(self, future_date, scenarios=100):
        """
        Forecast future factors based on the estimated Yule-Walker parameters.
        """
        future_date = pd.to_datetime(future_date, format='%Y-%m').to_period('M')

        # Debug: print de kolomwaarden en de huidige waarde van current_date
        # print("Columns in df_data:", self.df_data.columns)
        # print("Value of current_date before conversion:", self.df_data.columns[-1])

        # Haal de laatste datum op uit de dataset
        current_date = self.df_data.columns[-1]

        # Controleer of current_date al een Period is, en converteer indien nodig
        if not isinstance(current_date, pd.Period):
            current_date = pd.to_datetime(str(current_date), errors='coerce').to_period('M')
            if pd.isnull(current_date):
                raise ValueError(f"Invalid date format detected in current_date: {self.df_data.columns[-1]}")

        # print("Value of current_date after conversion:", current_date)

        if future_date <= current_date:
            raise ValueError("The future date must be later than the last date in the data.")

        # Debug: print het aantal maanden dat voorspeld gaat worden
        num_months = (future_date.year - current_date.year) * 12 + future_date.month - current_date.month
        # print(f"Number of months to forecast: {num_months}")

        # Controleer of num_months het verwachte aantal stappen is
        if num_months != scenarios:
            print(f"Warning: Number of months ({num_months}) does not match expected scenarios ({scenarios}).")

        phi = self.phi[1:].T
        intercept = self.phi[0]
        factors_forecast = []
        factors = self.factors.T[-1]

        for i in range(num_months):
            factors = np.dot(phi, factors) + intercept
            factors_forecast.append(factors)

            # Debug: Print de status van de voorspellingen
            # print(f"Forecast for month {i+1}: {factors}")

        return np.array(factors_forecast)