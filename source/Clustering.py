from sklearn.cluster import KMeans, DBSCAN, MeanShift, Birch
class Clustering():
    def __init__(self):
        pass

    def dDBSCAN(ep, min_sample, data):
        dDBSCAN = DBSCAN(eps = ep, min_samples = min_sample)
        return dDBSCAN.fit(data)
    
    def dKmeans(self, n, data):
        dKmeans = KMeans(n_clusters = n)
        return dKmeans.fit(data)
    
    def dMeanShift(data):
        dMeanShift = MeanShift()
        return dMeanShift.fit(data)
    
    def dBirch(branching_factor, n_clusters, threshold, data):
        dBirch = Birch(branching_factor= branching_factor, n_clusters=n_clusters, threshold=threshold)
        return dBirch.fit(data)