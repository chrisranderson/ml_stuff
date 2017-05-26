from sklearn.cluster import KMeans

def k_means(data, k=100):
  clusterer = KMeans(n_clusters=k, 
                     precompute_distances=True,
                     n_jobs=4)
  clusterer.fit(data)

  return clusterer
