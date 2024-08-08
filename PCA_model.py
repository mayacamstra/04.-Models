import numpy as np
from sklearn.decomposition import PCA

def apply_pca(std_data, num_factors):
    """
    Apply Principal Component Analysis (PCA) to the standardized data.

    Parameters:
    std_data (np.ndarray): The standardized data to which PCA will be applied.
    num_factors (int): The number of principal components to retain.

    Returns:
    np.ndarray: The principal components (factors).
    """
    pca = PCA(n_components=num_factors)
    factors = pca.fit_transform(std_data.T).T
    return factors
