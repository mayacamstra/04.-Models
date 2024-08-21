import numpy as np
import pandas as pd

def standardize(variables):
    """
    Standardize the variables by subtracting the mean and dividing by the standard deviation.

    Parameters:
    variables (np.ndarray or pd.DataFrame): The data to standardize.

    Returns:
    np.ndarray: The standardized data.
    """
    central = (variables - variables.mean())
    return central / central.std()

def RMSE(data: pd.DataFrame, estimation: pd.DataFrame):
    """
    Calculate the Root Mean Squared Error (RMSE) between the actual data and the estimation.

    Parameters:
    data (pd.DataFrame): The actual data.
    estimation (pd.DataFrame): The estimated data.

    Returns:
    np.ndarray: The RMSE values for each column.
    """
    df_errors = (estimation - data)
    df_rmse = ((df_errors) ** 2.0).mean(axis=0) ** 0.5
    return df_rmse

def calculate_r2(data: pd.DataFrame, estimation: pd.DataFrame):
    """
    Calculate the R^2 (coefficient of determination) between the actual data and the estimation.

    Parameters:
    data (pd.DataFrame): The actual data.
    estimation (pd.DataFrame): The estimated data.

    Returns:
    float: The R^2 value.
    """
    ss_res = ((data - estimation) ** 2).sum()
    ss_tot = ((data - data.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2

def log_likelihood(y_true, y_pred):
    """
    Bereken de log-likelihood voor een lineair model.

    Parameters:
    y_true (np.ndarray): De werkelijke waarden.
    y_pred (np.ndarray): De voorspelde waarden.

    Returns:
    float: De log-likelihood waarde.
    """
    residuals = y_true - y_pred
    n = len(y_true)
    sigma2 = np.var(residuals)
    log_like = -n/2 * (np.log(2 * np.pi * sigma2) + 1)
    return log_like

def calculate_aic_bic(y_pred, y_true, num_params):
    """
    Bereken AIC en BIC op basis van de log-likelihood.

    Parameters:
    y_pred (np.ndarray): De voorspelde waarden.
    y_true (np.ndarray): De werkelijke waarden.
    num_params (int): Het aantal parameters in het model.

    Returns:
    tuple: De AIC en BIC waarden.
    """
    log_like = log_likelihood(y_true, y_pred)
    n = len(y_true)
    aic = 2 * num_params - 2 * log_like
    bic = np.log(n) * num_params - 2 * log_like
    return aic, bic

def adjusted_r2(r2, n, p):
    """
    Bereken de adjusted R² waarde.

    Parameters:
    r2 (float): De R² waarde.
    n (int): Het aantal observaties.
    p (int): Het aantal voorspellers (aantal factoren).

    Returns:
    float: De adjusted R² waarde.
    """
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
