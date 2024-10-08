from sklearn.cross_decomposition import PLSRegression
import numpy as np
    
class PLSModel:
    def __init__(self, num_factors):
        self.num_factors = num_factors
        self.model = PLSRegression(n_components=num_factors)
        self.factors = None

    def fit_transform(self, X, Y):
        """
        Perform PLS regression and extract factors based on both X and Y.
        
        Parameters:
        X (np.ndarray): The input data (independent variables).
        Y (np.ndarray): The output data (dependent variables).

        Returns:
        np.ndarray: The latent factors.
        """
        # Perform PLS regression using both X and Y
        self.factors, _ = self.model.fit_transform(X, Y)

        # Debug: Check shape of factors
        print(f"Shape of PLS factors: {self.factors.shape}")
        
        # Ensure that the shape of the factors is correct (num_samples, num_factors)
        if self.factors.shape[1] != self.num_factors:
            raise ValueError(f"Expected {self.num_factors} factors, got {self.factors.shape[1]} factors")

        return self.factors