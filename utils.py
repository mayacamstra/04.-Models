import numpy as np
import pandas as pd

def standardize(variables):
    """
    Standardize the variables by subtracting the mean and dividing by the standard deviation.

    Parameters:
    variables (pd.DataFrame or np.ndarray): The data to standardize.

    Returns:
    pd.DataFrame or np.ndarray: The standardized data.
    """
    central = variables - variables.mean()
    return central / central.std()

def RMSE(data: pd.DataFrame, estimation: pd.DataFrame):
    """
    Calculate the Root Mean Squared Error (RMSE) between the actual data and the estimation.

    Parameters:
    data (pd.DataFrame): The actual data.
    estimation (pd.DataFrame): The estimated data.

    Returns:
    pd.Series: The RMSE values for each variable.
    """
    df_errors = estimation - data
    df_rmse = ((df_errors) ** 2.0).mean(axis=0) ** 0.5
    return df_rmse
