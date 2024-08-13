from sklearn.cross_decomposition import PLSRegression
import numpy as np

class PLSModel:
    def __init__(self, num_factors):
        self.num_factors = num_factors
        self.model = PLSRegression(n_components=num_factors)
        self.factors = None

    def fit_transform(self, data):
        self.factors, _ = self.model.fit_transform(data, np.zeros(data.shape[0]))
        return self.factors

    def transform(self, data):
        return self.model.transform(data)