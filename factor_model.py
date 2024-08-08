import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import r2_score
from PCA_model import apply_pca
from utils import standardize
from datetime import datetime

class DynamicFactorModel:
    def __init__(self, df_data, num_factors):
        self.df_data = df_data
        self.num_factors = num_factors
        self.std_data = standardize(df_data.values.T).T
        self.factors = None
        self.phi = None
        self.B_mat = None
        self.model_ena = None

    def apply_pca(self):
        self.factors = apply_pca(self.std_data.T, self.num_factors)

    def yw_estimation(self):
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
    
    def autoregression(self, data_train_reg, fac_train, beta_const):
        X = data_train_reg.T
        Y = (self.std_data.T - np.dot(fac_train, self.B_mat.T) - beta_const).T

        if X.shape[1] != Y.shape[1]:
            Y = Y[:, :X.shape[1]]

        if X.shape[0] != Y.shape[0]:
            Y = Y[:X.shape[0], :]

        Y = np.matrix(Y)
        X = np.matrix(X)

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Het aantal rijen in X en Y moet gelijk zijn na transponeren")

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
            raise ValueError("De toekomstige datum moet later zijn dan de laatste datum in de data.")
        num_months = (future_date.year - current_date.year) * 12 + future_date.month - current_date.month
        
        phi = self.phi[1:].T
        intercept = self.phi[0]
        factors_forecast = []
        factors = self.factors.T[-1]
        
        for _ in range(num_months):
            factors = np.dot(phi, factors) + intercept
            factors_forecast.append(factors)
        
        return np.array(factors_forecast)
