
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
    def fit(self, x, dim, epochs = 1000, alpha_initial = 0.1, alpha_decay = None, neighborhood = None):
        X = np.array([normalize(x_i) for x_i in x])

        if neighborhood is None:
            neighborhood = neighborhood_linear(dim/2, epochs/2)
        
        if alpha_decay is None:
            alpha_decay = linear_decay(alpha_initial, epochs)

        np.random.shuffle(X)
        self.network = [] # Lista de {x, y, weights}
        c = 0
        for i in range(0, dim):
            for j in range(0, dim):
                self.network.append({"x": i, "y": j, "weights": np.array(X[c])})    # Inicializo con pesos aleatorios
                c += 1

        for t in range(1, epochs):
            np.random.shuffle(X)
            for x_i in X:
                distances = [np.dot(x_i, n["weights"]) for n in self.network ]
                winner = self.network[np.argmin(distances)]
                for n in self.network:
                    n["weights"] = n["weights"] + neighborhood(t, grid_distance(n, winner)) * alpha_decay(t) * (x_i - n["weights"]) 
    
    def predict(self, x):
        x = normalize(x)
        distances = [np.dot(x, n["weights"]) for n in self.network ]
        return np.argmin(distances)


