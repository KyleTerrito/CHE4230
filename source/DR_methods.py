from sklearn.decomposition import PCA
class DR():
    def __init__(self):
        pass

    def dPCA(self, n, data):
        DPCA  = PCA(n_components = n)
        return DPCA.fit_transform(data)