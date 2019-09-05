import numpy as np
from DecisionTree import DecisionTree, DecisionTreeLeaf, dot_string

# Metric functions applied to frequency arrays
ENTROPY = lambda fs: -sum([0 if f == 0 else f * np.log2(f) for f in fs])
GINI_INDEX = lambda fs: 1 - sum([f*f for f in fs])
METRIC_FUNCTIONS = { 'gain': ENTROPY, 'gini': GINI_INDEX }

def metric_apply(klasses, metric_f):
    _, unique_counts = np.unique(klasses, return_counts=True)
    rel_freqs = unique_counts / len(klasses)
    return metric_f(rel_freqs)

def gain(x, y, attr_i, metric_f):
    all_attr_values = x.T[attr_i]
    attr_values, attr_values_count = np.unique(all_attr_values, return_counts=True)

    gain = metric_f(y)
    total = len(y)

    for attr_value, attr_value_count in zip(attr_values, attr_values_count):
        filtered_by_attribute = np.where(all_attr_values == attr_value)
        gain -= ((attr_value_count/total) * metric_apply(y[filtered_by_attribute], metric_f))
    return gain


def max_gain_attribute(x, y, usable_attrs, *, metric_f):
    # Get index of attribute with maximun gain given the metric
    gains = []
    for i in usable_attrs:
        g = gain(x, y, i, metric_f)
        gains.append(g)
    
    return usable_attrs[np.argmax(gains)]

def create_decision_tree(x, y, usable_attrs = None, *, metric_f_name):
    metric_f = METRIC_FUNCTIONS[metric_f_name]
    x = np.array(x)
    y = np.array(y)

    if x.shape[0] != y.shape[0]:
        raise f"X and Y row count mismatch. X: {x.shape[0]} - Y: {y.shape[0]}"

    if usable_attrs is None:
        usable_attrs = np.arange(x.shape[1])

    # print('Usable attrs: ', usable_attrs)

    # If there is only one class in the set, create a leaf node with that class.
    if len(np.unique(y)) == 1:
        return DecisionTreeLeaf(y[0])
    
    # Get attribute with maximum information 
    best_attr = max_gain_attribute(x, y, usable_attrs, metric_f = metric_f)
    print('Max gain attr: ', best_attr)

    usable_attrs = usable_attrs[usable_attrs != best_attr]
    
    children = {}
    all_attr_values = x.T[best_attr]
    attr_unique_values = np.unique(all_attr_values)

    # Create children -> subtrees slicing the dataset with the attribute that is set
    for v in attr_unique_values:
        # Only take rows where attr = v
        v_rows = np.where(all_attr_values == v)
        x_slice = x[v_rows]
        y_slice = y[v_rows]
        children[v] = create_decision_tree(x_slice, y_slice, usable_attrs = usable_attrs, metric_f_name = metric_f_name)
    
    root = DecisionTree(best_attr, children=children)
    return root

if __name__ == '__main__':
    PRONOSTICO, TEMPERATURA, HUMEDAD, VIENTO = 0, 1, 2, 3
    SOL, NUBLADO, LLUVIOSO = 0, 1, 2
    FRIO, TEMPLADO, CALIDO = 0, 1, 2
    ALTA, NORMAL = 0, 1
    FUERTE, DEBIL = 0, 1
    NO, SI = 0, 1

    tree = create_decision_tree(
        [
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
        ], 
        [NO, NO, SI, SI, SI, NO, SI, NO, SI, SI, SI, SI, SI, NO],
        metric_f_name = 'gain'
    )

    print(dot_string(tree, feature_names = ['PRONOSTICO', 'TEMPERATURA', 'HUMEDAD', 'VIENTO'], feature_values = [
        ['SOL', 'NUBLADO', 'LLUVIOSO'],
        ['FRIO', 'TEMPLADO', 'CALIDO'],
        ['ALTA', 'NORMAL'],
        ['FUERTE', 'DEBIL']
    ], class_names=['NO', 'SI']))


# Poda
# Mínimo de observaciones para dividir un nodo -> como está ahora se puede hacer
# Máxima profundidad del árbol (vertical) -> pasar altura del árbol en la recursión
# Máximo número de atributos a considerar para la ramificación -> como está ahora se puede hacer
# Máximo número de nodos hoja -> ?
# Gini < u -> ?
#
# Al expandir, etiquetar el nodo con la clase mas frecuente
