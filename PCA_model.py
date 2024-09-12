import numpy as np
from sklearn.decomposition import PCA

def apply_pca(std_data, num_factors, return_model=False):
    """
    Apply Principal Component Analysis (PCA) to the standardized data.

    Parameters:
    std_data (np.ndarray): The standardized data to which PCA will be applied.
    num_factors (int): The number of principal components to retain.
    return_model (bool): Whether to return the PCA model object along with the factors.

    Returns:
    np.ndarray: The principal components (factors).
    PCA: (optional) The PCA model, only returned if return_model is True.
    """
    pca = PCA(n_components=num_factors)
    factors = pca.fit_transform(std_data)  # Make sure 'std_data' is passed here
    
    if return_model:
        return factors, pca  # Return both the factors and the PCA model
    return factors  # Return only the factors if return_model is False
