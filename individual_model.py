from factor_model import DynamicFactorModel
from utils import standardize, RMSE
import pandas as pd

class IndividualModel:
    def __init__(self, df_data, num_factors=9, method='PCA'):
        self.model = DynamicFactorModel(df_data, num_factors, method)

    def train(self, data_train, data_train_reg):
        self.B_matrix, self.C_matrix, self.r2_insample, self.beta_const = self.model.dfm_fit_pcayw(data_train, data_train_reg)
        return self.B_matrix, self.C_matrix, self.r2_insample, self.beta_const

    def predict(self, data_predict, data_predict_reg):
        factor_forecast = self.model.factor_forecast(data_predict.columns[-1].to_timestamp(), scenarios=100)
        predictions = self.model.enet_predict(factor_forecast.T)
        return predictions