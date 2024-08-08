import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import r2_score

def standardize(variables):
    """
    Standardize the variables by subtracting the mean and dividing by the standard deviation.

    Parameters:
    variables (pd.DataFrame or np.ndarray): The data to standardize.

    Returns:
    pd.DataFrame or np.ndarray: The standardized data.
    """
    central = (variables - variables.mean())
    return central / central.std()

def enet_fit(data_train, fac_train):
    """
    Fit a MultiTaskElasticNet model to the training data.

    Parameters:
    data_train (np.ndarray): The training data.
    fac_train (np.ndarray): The training factors.

    Returns:
    np.ndarray: The coefficients of the fitted model.
    float: The in-sample R-squared value.
    np.ndarray: The intercept of the fitted model.
    """
    model_ena = MultiTaskElasticNetCV(cv=5)
    model_ena.fit(fac_train, data_train)
    B_mat = model_ena.coef_
    x_hat = model_ena.predict(fac_train)
    intercept = model_ena.intercept_
    r2_insample = r2_score(data_train, x_hat)
    return B_mat, r2_insample, intercept

def enet_predict(model_ena, fac_predict):
    """
    Use the fitted MultiTaskElasticNet model to make predictions.

    Parameters:
    model_ena (MultiTaskElasticNetCV): The fitted model.
    fac_predict (np.ndarray): The factors for prediction.

    Returns:
    np.ndarray: The predicted values.
    """
    return model_ena.predict(fac_predict)
