from KNNClassifier import KNNClassifier
from random import randint
import numpy as np

def group_examples(X, assignments, k):
    groups = [[] for _ in range(k)] # k clusters
    for (i,x) in zip(assignments, X):
        groups[i].append(x)
    return groups

def centroid(xs):
    return np.mean(xs, axis=0)

def recalculate_assignments(X, centroids):
    knn = KNNClassifier(K = 1) # Look for the nearest neighbor
    knn.fit(centroids, [i for i in range(len(centroids))]) # Cada centroide tiene su indice como clase
    assignments = [knn.predict(x) for x in X]   # Me devuelve los indices de cada centroide mas cercano
    return assignments


def kmeans(X, k):
    size = len(X)
    assignments = [randint(0,k-1) for _ in range(size)] # Initial assignment -- random
    loop = True
    while loop:
        assignments_prev = assignments
        groups = group_examples(X, assignments, k)
        centroids = [centroid(g) for g in groups]
        assignments = recalculate_assignments(X, centroids)
        loop = any(map(lambda pair: pair[0] != pair[1] , zip(assignments, assignments_prev))) # Loop if anything changed
    return list(zip(X, assignments))


    

