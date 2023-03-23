from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
import umap
import pacmap


class DR():
    def __init__(self):
        pass

    def dPCA(self, n, data):
        DPCA  = PCA(n_components = n)
        return DPCA.fit_transform(data)
    
    def dtSNE(self, n, data):
        dtSNE = TSNE(n_components= n)
        return dtSNE.fit_transform(data)
    
    def dFastICA(self, n, data):
        dFastICA = FastICA(n_components= n)
        return dFastICA.fit_transform(data)
    
    def dPacMap(self, n, data):
        dpacmap = pacmap.PaCMAP(n_components= n, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
        return dpacmap.fit_transform(data)

    def dumap(self, data):
        dumap = umap.UMAP()
        return dumap.fit_transform(data)