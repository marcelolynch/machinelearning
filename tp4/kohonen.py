
import numpy as np
def neighborhood_linear(initial_radius, decay_time):
    return lambda t, dist: 1 if dist < (initial_radius - (initial_radius / decay_time) * t) else 0

def linear_decay(initial, decay_time):
    return lambda t: (initial - (initial / decay_time) * t)


def grid_distance(n1, n2):
    return abs(n1["x"] - n2["x"]) + abs(n1["y"] - n2["y"])

def normalize(x):
    norm = np.linalg.norm(x)
    x = np.array(x)
    return x / np.linalg.norm(x)

class Kohonen:
    def fit(self, X, dim, epochs = 1000, alpha = None, neighborhood = None):
        X = np.array([normalize(x_i) for x_i in X])     # Normalizar los ejemplos

        if neighborhood is None:
            neighborhood = neighborhood_linear(dim/2, epochs/2)
        
        if alpha is None:
            alpha = linear_decay(0.1, epochs)

        self.network = []   # Cada nodo es un dict {x, y, weights}
        c = 0

        np.random.shuffle(X)
        for i in range(0, dim):
            for j in range(0, dim):
                # Inicializo con pesos tomando elementos aleatorios del conjunto de entrenamiento
                self.network.append({"x": i, "y": j, "weights": np.array(X[c])})    
                c += 1

        for t in range(1, epochs):
            np.random.shuffle(X)    # Voy a recorrerlos todos en un orden aleatorio
            for x_i in X:
                distances = [np.dot(x_i, n["weights"]) for n in self.network ]
                winner = self.network[np.argmin(distances)]
                for n in self.network:
                    n["weights"] = n["weights"] + neighborhood(t, grid_distance(n, winner)) * alpha(t) * (x_i - n["weights"]) 
    
    def predict(self, x):
        x = normalize(x)
        distances = [np.dot(x, n["weights"]) for n in self.network ]
        return np.argmin(distances)


