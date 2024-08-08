import statsmodels.api as sm

def yw_estimation(factors):
    """
    Perform Yule-Walker estimation on the factors to fit a Vector Autoregression (VAR) model.

    Parameters:
    factors (np.ndarray): The principal components (factors) from PCA.

    Returns:
    np.ndarray: The estimated parameters from the Yule-Walker estimation.
    """
    model = sm.tsa.VAR(factors.T)
    results = model.fit(1)
    phi = results.params
    return phi
