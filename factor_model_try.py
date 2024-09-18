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
        Perform Yule-Walker estimation on the factors to fit a VAR model specifically for the 5 factors.
        """
        # We gebruiken alleen de 5 factoren om de VAR te schatten
        factors_transposed = self.factors  # Vorm is nu (num_factors, num_time_points), dus (5, 300)

        # Debug: Controleer de vorm van de getransponeerde factorenmatrix
        print(f"Shape of factors after transposition: {factors_transposed.shape}")

        # Nu passen we het VAR-model toe op de getransponeerde factoren om de Phi-matrix te krijgen
        model = sm.tsa.VAR(factors_transposed)
        results = model.fit(1)  # We schatten een VAR(1) model

        # Sla alleen de relevante autoregressieve parameters op
        self.phi = results.params  # Vorm van phi is nu (num_factors + 1, num_factors), dus (6, 5)
        
        # Debug: Print de nieuwe Phi-matrix en de vorm
        print(f"Yule-Walker estimation (phi matrix):\n{self.phi}")
        print(f"Shape of phi matrix: {self.phi.shape}")

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

    def factor_forecast(self, num_steps=1):
        """
        Forecast future factors based on the estimated Yule-Walker parameters, now focusing on 5 factors.
        
        Parameters:
        num_steps (int): Number of steps to forecast (default is 1).

        Returns:
        np.ndarray: Predicted factors for the next time step.
        """
       # Start met de laatste rij van de factoren (meest recente factoren)
        current_factors = self.factors[-1]  # Dit is een rij van vorm (num_factors,)

        # Debug: Print de huidige vorm van current_factors
        print(f"Shape of current_factors: {current_factors.shape}")

        # Phi moet nu de vorm (5, 5) hebben voor de autoregressieve parameters
        phi = self.phi[1:].T  # De eerste rij is de intercept, dus we gebruiken vanaf de tweede rij

        # Debug: Print de vorm van de Phi-matrix
        print(f"Shape of phi matrix used for forecasting: {phi.shape}")

        # Intercept is de eerste rij van de Phi-matrix (vorm is (5,))
        intercept = self.phi[0]  

        # Debug: Print de vorm van de intercept
        print(f"Shape of intercept: {intercept.shape}")

        predicted_factors = []

        # Voorspel voor het aantal gewenste stappen (meestal 1 stap)
        for _ in range(num_steps):
            # Bereken de volgende factoren door de huidige factoren met Phi te vermenigvuldigen
            next_factors = np.dot(phi, current_factors) + intercept

            # Debug: Print de vorm van next_factors
            print(f"Shape of next_factors: {next_factors.shape}")

            predicted_factors.append(next_factors)
            
            # Update de huidige factoren voor de volgende iteratie (indien meerdere stappen nodig zijn)
            current_factors = next_factors

        # Return de voorspelling als een 2D-array (num_steps, num_factors)
        return np.array(predicted_factors).reshape(num_steps, -1)
