from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
class DR():
    def __init__(self):
        pass

    def dPCA(self, n, data):
        DPCA  = PCA(n_components = n)
        return DPCA.fit_transform(data)
    
    def dtSNE(self, n, data):
        dtSNE = TSNE(n_components= n)
        return dtSNE.fit_transform(data)
    
    def dDBSCAN(self, data):
        dDBSCAN = DBSCAN()
        return dDBSCAN.fit(data)
    