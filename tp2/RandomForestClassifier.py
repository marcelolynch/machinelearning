import numpy as np
from DecisionTree import DecisionTree, DecisionTreeLeaf, dot_string
from metrics import score

def entropy(klasses):
    _, unique_counts = np.unique(klasses, return_counts=True)
    rel_freqs = unique_counts / len(klasses)
    return -np.sum(np.multiply(rel_freqs, np.log2(rel_freqs)))

# Mmmm tengo dudas de como hacer las otras funciones de información
def gain(x, y, attr_i):
    all_attr_values = x.T[attr_i]
    attr_values, attr_values_count = np.unique(all_attr_values, return_counts=True)

    gain = entropy(y)
    total = len(y)

    for attr_value, attr_value_count in zip(attr_values, attr_values_count):
        # print(f'Attribute value = {attr_value}. |Sattr=v| = {attr_value_count}')
        filtered_by_attribute = np.where(all_attr_values == attr_value)
        gain -= ((attr_value_count/total) * entropy(y[filtered_by_attribute]))
    return gain

def max_information_attribute(x, y, usable_attrs, *, info_f):
    # Get index of attribute with maximun information function
    gains = []
    for i in usable_attrs:
        g = gain(x, y, i)
        gains.append(g)
    
    return usable_attrs[np.argmax(gains)]

INFORMATION_FUNCTIONS = { 'gain': gain }

def create_decision_tree(x, y, usable_attrs = None, *, information_f):
    info_f = INFORMATION_FUNCTIONS[information_f]
    x = np.array(x)
    y = np.array(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X and Y row count mismatch. X: {x.shape[0]} - Y: {y.shape[0]}")

    if usable_attrs is None:
        usable_attrs = np.arange(x.shape[1])

    # print('Usable attrs: ', usable_attrs)

    # If there is only one class in the set, create a leaf node with that class.
    if len(np.unique(y)) == 1:
        return DecisionTreeLeaf(y[0])
    
    # Get attribute with maximum information 
    best_attr = max_information_attribute(x, y, usable_attrs, info_f = info_f)
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
        children[v] = create_decision_tree(x_slice, y_slice, usable_attrs = usable_attrs, information_f = information_f)
    
    root = DecisionTree(best_attr, children=children)
    return root

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

    tree = create_decision_tree(x = train_x, y = train_y, information_f = 'gain')

    print(dot_string(tree, feature_names = ['PRONOSTICO', 'TEMPERATURA', 'HUMEDAD', 'VIENTO'], feature_values = [
        ['SOL', 'NUBLADO', 'LLUVIOSO'],
        ['FRIO', 'TEMPLADO', 'CALIDO'],
        ['ALTA', 'NORMAL'],
        ['FUERTE', 'DEBIL']
    ], class_names=['NO', 'SI']))

    score(tree, train_x, train_y, ['NO', 'SI'], confusion_matrix=True)


# Poda
# Mínimo de observaciones para dividir un nodo -> como está ahora se puede hacer
# Máxima profundidad del árbol (vertical) -> pasar altura del árbol en la recursión
# Máximo número de atributos a considerar para la ramificación -> como está ahora se puede hacer
# Máximo número de nodos hoja -> ?
# Gini < u -> ?
#
# Al expandir, etiquetar el nodo con la clase mas frecuente
