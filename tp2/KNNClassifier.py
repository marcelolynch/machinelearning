import numpy as np
from heapq import heappush, heappop

EUCLIDEAN_DISTANCE = lambda a, b: np.linalg.norm(a - b)
MANHATTAN_DISTANCE = lambda a, b: sum(np.abs(a - b))

# Sino podemos poner que siempre use la distancia de Minkowski y pasamos q como parámetro
# https://www.saedsayad.com/k_nearest_neighbors.htm
# Interesante también ver el tema de distancia de Hamming para cuando las variables son categóricas

DISTANCE_FUNCTIONS = { 'euclidean': EUCLIDEAN_DISTANCE, 'manhattan': MANHATTAN_DISTANCE, 'one': lambda a, b: 1 }

class KNNClassifier():
    def __init__(self, *, K, distance_f = 'euclidean'):
        self.K = K
        self.train_x = None
        self.train_y = None
        self.distance_f = DISTANCE_FUNCTIONS[distance_f]

    def train(self, x, y):
        if len(x) != len(y):
            raise ValueError(f'Sizes of x ({len(x)}) and y ({len(y)}) mismatch')
        
        x = np.array(x)
        y = np.array(y)

        if self.train_x is None:
            self.train_x = x
            self.train_y = y
        else:
            self.train_x = np.append(self.train_x, x, axis=0)
            self.train_y = np.append(self.train_y, y)

    # TODO: cell-index method si nos da la gana
    def predict(self, x_i):
        if len(self.train_x) < self.K:
            raise ValueError(f'Cannot make a prediction with less than K examples ({len(self.train_x)}) in the training set')
        
        x_i = np.array(x_i)

        k_nearest = []
        # Push K elements
        for j in range(self.K):
            x_j = self.train_x[j]
            # Negative distance for a 'max heap' of distances to the input vector
            heappush(k_nearest, (-self.distance_f(x_i, x_j), j)) 
        
        # Push and pop one by one until the train set runs out of examples
        for j in range(self.K, len(self.train_x)):
            x_j = self.train_x[j]
            heappush(k_nearest, (-self.distance_f(x_i, x_j), j))
            heappop(k_nearest)

        class_sums = {}
        # K examples remaining are the nearest ones to the input vector
        for negative_dist, i in k_nearest:
            klass = self.train_y[i]
            if klass not in class_sums:
                class_sums[klass] = 0 
            
            # Return immediately if vector is the same as one in the train set
            if negative_dist == 0:
                return klass

            class_sums[klass] += 1/(-negative_dist)

        # print(class_sums)
        return max(class_sums, key=class_sums.get)

# Crude example
# from sklearn import datasets
# from random import shuffle
# iris = datasets.load_iris()
# indexes = np.arange(len(iris.data))
# np.random.shuffle(indexes)

# x = iris.data[indexes]
# y = iris.target[indexes]

# knnc = KNNClassifier(K = 100)

# knnc.train(x[:100], y[:100])

# GUESS_I = 134
# p = knnc.predict(x[GUESS_I])
# print(f'Prediction: {p}, Actual value: {y[GUESS_I]}')