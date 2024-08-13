import numpy as np
from sklearn.decomposition import PCA

class PCAModel:
    def __init__(self, num_factors):
        self.num_factors = num_factors
        self.model = PCA(n_components=num_factors)
        self.factors = None

    def fit_transform(self, std_data):
        self.factors = self.model.fit_transform(std_data.T).T
        return self.factors

    def transform(self, std_data):
        return self.model.transform(std_data.T).T