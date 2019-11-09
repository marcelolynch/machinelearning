import numpy as np

def create_cluster_hierarchy(xs, distance_measure = 'complete_link'):
  clusters = [Cluster([x_i], order=i) for i, x_i in enumerate(xs)]

  c_order = len(clusters)
  while (len(clusters) > 1):
    # Get distance between each pair of clusters
    pair_dists = []
    for i in range(len(clusters)):
      for j in range(i+1, len(clusters)):
        ci = clusters[i]
        cj = clusters[j]
        pair_dists.append((i, j, ci.distance_to(cj, distance_measure)))
    
    # Replace the two closest clusters with a new one that is the result of merging both.
    # c_order is the order in which the clusters were created.
    min_dist = min(pair_dists, key = lambda t: t[2])
    to_delete = sorted(min_dist[:2], reverse = True)
    ci = clusters.pop(to_delete[0])
    cj = clusters.pop(to_delete[1])
    all_elems = np.concatenate((ci.elems, cj.elems))
    clusters.append(Cluster(all_elems, left_cluster = ci, right_cluster = cj, cluster_distance = min_dist[2], order = c_order))
    c_order += 1

  # We end with just one cluster resulting from merging all the others sequentially
  return clusters[0]

DISTANCE_MEASURES = {
  'complete_link': lambda a, b:  np.max(abs(a-b)),
  'single_link': lambda a, b: np.min(abs(a-b)),
  'average_link': lambda a, b: np.mean(abs(a-b)),
  'centroid': lambda a, b: abs(np.mean(a, axis=0)-np.mean(b, axis=0)),
}

class Cluster():
  def __init__(self, elems, *, left_cluster = None, right_cluster = None, cluster_distance = None, order):
    if (len(elems) == 1): # Leaf
      assert left_cluster is None and right_cluster is None, 'Should not define child cluster for leaf node'
      assert cluster_distance is None, 'Should not define cluster distance for leaf node'
      self.cluster_distance = 0
    else:
      assert left_cluster is not None and right_cluster is not None, 'Should define both child clusters for non-leaf node'
      assert cluster_distance is not None, 'Should define cluster distance for non-leaf node'
      self.left_cluster = left_cluster
      self.right_cluster = right_cluster
      self.cluster_distance = cluster_distance

    self.elems = np.array(elems) # TODO: normalización
    self.order = order

  def is_leaf(self):
    return len(self.elems) == 1

  def distance_to(self, cluster, measure = 'complete_link'):
    return DISTANCE_MEASURES[measure](self.elems, cluster.elems)

  def __repr__(self):
    return f"C: {str(self.elems)} - cl_dist: {self.cluster_distance}"


# Returns a matrix with the format that matplot wants to draw the dendrogram
def get_linkage_matrix(cluster):
  Z = []
  clusters = [cluster]
  while (len(clusters) > 0):
    new_clusters = []
    for c in clusters:
      row = np.array([c.left_cluster.order, c.right_cluster.order, c.cluster_distance + 1, 1], dtype=np.float)
      Z.append(row)

      if not c.left_cluster.is_leaf():
        new_clusters.append(c.left_cluster)
      if not c.right_cluster.is_leaf():
        new_clusters.append(c.right_cluster)
    
    clusters = new_clusters
  return np.array(sorted(Z, key = lambda r: r[2]))

# Sample usage

# DD = create_cluster_hierarchy([[1, 2, 3], [3, 2, 5], [4, 5, 7], [4, 6, 7], [40, 60, 70]], distance_measure='single_link')
# Z = get_linkage_matrix(DD)
# print('--- Z ---')
# print(Z)

# from scipy.cluster.hierarchy import dendrogram
# import matplotlib.pyplot as plt
# plt.figure()
# dn = dendrogram(Z)
# plt.show()