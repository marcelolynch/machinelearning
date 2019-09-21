import numpy as np
from DecisionTreeBuilder import create_decision_tree
from model_selector import Bootstrap
from collections import Counter
from metrics import score


class RandomForestClassifier:
    def __init__(self, n_trees, bootstrap_size = 1, metric_f = 'gain'):
        self.n_trees = n_trees
        self.metric_f = metric_f
        self.bootstrap_size = bootstrap_size
        self.trees = []

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        bootstrapper = Bootstrap(self.n_trees, self.bootstrap_size)
        for selected, _ in bootstrapper.split(x):
            x_bootstrapped = x[selected]
            y_bootstrapped = y[selected]
            self.trees.append(create_decision_tree(x_bootstrapped, y_bootstrapped, metric_f_name = self.metric_f))       

    def predict(self, x):
        predictions = [t.predict(x) for t in self.trees]
        most_common, _ = Counter(predictions).most_common(1)[0]
        return most_common




if __name__ == '__main__':
    PRONOSTICO, TEMPERATURA, HUMEDAD, VIENTO = 0, 1, 2, 3
    SOL, NUBLADO, LLUVIOSO = 0, 1, 2
    FRIO, TEMPLADO, CALIDO = 0, 1, 2
    ALTA, NORMAL = 0, 1
    FUERTE, DEBIL = 0, 1
    NO, SI = 0, 1

    train_x = [
        [SOL, CALIDO, ALTA, DEBIL], 
        [SOL, CALIDO, ALTA, FUERTE], 
        [NUBLADO, CALIDO, ALTA, DEBIL],
        [LLUVIOSO, TEMPLADO, ALTA, DEBIL],
        [LLUVIOSO, FRIO, NORMAL, DEBIL],
        [LLUVIOSO, FRIO, NORMAL, FUERTE],
        [NUBLADO, FRIO, NORMAL, FUERTE],
        [SOL, TEMPLADO, ALTA, DEBIL],     
        [SOL, FRIO, NORMAL, DEBIL], 
        [LLUVIOSO, TEMPLADO, NORMAL, DEBIL], 
        [SOL, TEMPLADO, NORMAL, FUERTE], 
        [NUBLADO, TEMPLADO, ALTA, FUERTE],
        [NUBLADO, CALIDO, NORMAL, DEBIL],
        [LLUVIOSO, TEMPLADO, ALTA, FUERTE],   
    ]

    train_y = [NO, NO, SI, SI, SI, NO, SI, NO, SI, SI, SI, SI, SI, NO]

    rf = RandomForestClassifier(10, 1)
    rf.fit(train_x, train_y)

    score(rf, train_x, train_y, ['NO', 'SI'], confusion_matrix=True)
