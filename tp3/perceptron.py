import numpy as np

class Perceptron:
    def __init__(self, no_of_inputs, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.weights = (-1 + (2 * np.random.rand(no_of_inputs + 1))) * 0.1

    def get_weighted_output(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

    def predict(self, inputs):
        return np.sign(self.get_weighted_output(inputs))

    def train(self, x, y, threshold=-1):
        errors = 1
        epochs = threshold
        while errors > 0 and epochs != 0:
            epochs -= 1
            errors = 0
            for inputs, label in zip(x, y):
                prediction = self.predict(inputs)
                if prediction != label:
                    errors += 1
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)