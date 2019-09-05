# TODO: rename file 
import numpy as np
from DecisionTree import DecisionTree, DecisionTreeLeaf, dot_string

def entropy(klasses):
    _, unique_counts = np.unique(klasses, return_counts=True)
    total = len(klasses)
    H = 0
    # TODO: capaz se puede hacer con magia de numpy
    for c in unique_counts:
        rel_freq = c/total
        H -= (rel_freq * np.log2(rel_freq))
    return H

def max_gain_attribute(x, y, usable_attrs):
    total = len(y)
    gains = []
    # For each attribute, calculate gain
    # TODO: information function
    H = entropy(y)
    print('Entropy: ', H)
    for i in usable_attrs:
        all_attr_values = x.T[i]

        # Problema: me puede pasar que al hacer los cortes sobre las rows pierda los valores posibles de algun atributo.
        # Que hacemos? Lo cargamos aparte? 
        # Numericamente no cambia nada pero haría que no haya una rama del árbol de decisión en ese lugar? No, anda bien
        attr_values, attr_values_count = np.unique(all_attr_values, return_counts=True)

        gain = H # TODO: nombre???
        for attr_value, attr_value_count in zip(attr_values, attr_values_count):
            # print(f'Attribute value = {attr_value}. |Sattr=v| = {attr_value_count}')
            filtered_by_attribute = np.where(all_attr_values == attr_value)
            gain -= ((attr_value_count/total) * entropy(y[filtered_by_attribute]))

        gains.append(gain)
    
    return usable_attrs[np.argmax(gains)]

def create_decision_tree(x, y):
    #TODO: check sizes match
    x = np.array(x)
    y = np.array(y)
    all_attrs = np.array(list(range(x.shape[1]))) # Wat
    return create_decision_subtree(x, y, all_attrs)

    
def create_decision_subtree(x, y, usable_attrs):
    print('Usable attrs: ', usable_attrs)

    if len(np.unique(y)) == 1:
        return DecisionTreeLeaf(y[0])
    
    # Get attribute with maximum information 
    best_attr = max_gain_attribute(x, y, usable_attrs)
    print('Max gain attr: ', best_attr)

    usable_attrs = usable_attrs[usable_attrs != best_attr]
    
    children = {}
    all_attr_values = x.T[best_attr]
    attr_unique_values = np.unique(all_attr_values)

    # Create children
    for v in attr_unique_values:
        # Only take rows where attr = v
        v_rows = np.where(all_attr_values == v)
        x_slice = x[v_rows]
        y_slice = y[v_rows]
        children[v] = create_decision_subtree(x_slice, y_slice, usable_attrs=usable_attrs)
    
    # Create tree with that attribute as root
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
        [NO, NO, SI, SI, SI, NO, SI, NO, SI, SI, SI, SI, SI, NO]
    )

    print(dot_string(tree, feature_names = ['PRONOSTICO', 'TEMPERATURA', 'HUMEDAD', 'VIENTO'], class_names=['NO', 'SI']))
