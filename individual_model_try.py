from factor_model import DynamicFactorModel
from utils import standardize, RMSE
import pandas as pd
import numpy as np

class IndividualModel:
    def __init__(self, df_data, num_factors=9, method='PCA'):
        # Zorg ervoor dat df_data een Pandas DataFrame is, zodat 'values' kan worden gebruikt
        if isinstance(df_data, np.ndarray):
            # Zet de numpy-array om naar een DataFrame als dat nog niet gebeurd is
            df_data = pd.DataFrame(df_data)
        
        # Initialiseer het DynamicFactorModel met de df_data
        self.model = DynamicFactorModel(df_data, num_factors, method)

    def train(self, data_train, data_train_reg):
        # Train het model met de trainingsdata en regressie data
        self.B_matrix, self.C_matrix, self.r2_insample, self.beta_const = self.model.dfm_fit(data_train, data_train_reg)
        return self.B_matrix, self.C_matrix, self.r2_insample, self.beta_const

    def predict(self, data_predict, data_predict_reg):
        # Voorspel de variabelen met de gefit factors
        factor_forecast = self.model.factor_forecast(data_predict.columns[-1].to_timestamp(), scenarios=100)
        predictions = self.model.enet_predict(factor_forecast.T)
        return predictions
