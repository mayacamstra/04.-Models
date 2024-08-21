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

def calculate_aic_bic(y_hat, y_true, num_params):
    n = len(y_true)
    residuals = y_true - y_hat
    sse = np.sum(residuals**2)
    aic = n * np.log(sse/n) + 2 * num_params
    bic = n * np.log(sse/n) + num_params * np.log(n)
    return aic, bic