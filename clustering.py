from sklearn.cluster import KMeans

def k_means(data, k=100):
  clusterer = KMeans(n_clusters=k, precompute_distances=True)
  clusterer.fit(data)

  return clusterer
