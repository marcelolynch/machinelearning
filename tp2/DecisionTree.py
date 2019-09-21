import random

class DecisionTree():
    def __init__(self, attr, best_class, *, children = {}):
        self.attr = attr
        self.leaf_value = best_class
        self.children = children # Dict attr_value -> DecisionTree

    def predict(self, x):
        if x[self.attr] not in self.children: 
            # No conozco estos ejemplos, así que soy como 'hoja':
            # doy lo mas frecuente sabiendo lo que se hasta ahora
            return self.leaf_value

        subtree = self.children[x[self.attr]]
        return subtree.predict(x)

    def is_leaf(self):
        return False

    def size(self):
        return 1 + sum([c.size() for c in self.children.values()])
            

class DecisionTreeLeaf(DecisionTree):
    def __init__(self, leaf_value):
        self.leaf_value = leaf_value
    
    def predict(self, x):
        return self.leaf_value

    def is_leaf(self):
        return True
    
    def size(self):
        return 1
        
def dot_string(tree, *, feature_names, feature_values, class_names, options=""):
    return 'digraph G {\n '+ options +'\n'  + 'root [label="", shape="doublecircle", style=solid, width=0.2]\n' + dot_string_rec('root', tree, feature_names = feature_names, feature_values = feature_values, class_names = class_names) + '}'

def dot_string_rec(curr, tree, *, feature_names, feature_values, class_names):
    dot_str = ''

    if tree.is_leaf():
        klass = class_names[tree.leaf_value]
        c = curr.replace('"', '')
        node_name = f'{random.randint(0,100000)}'
        return ''
        #return f'    {curr} -> {node_name}\n    {node_name} [label="<{klass}>"]\n'

    f_name = feature_names[tree.attr]
    value_names = feature_values[tree.attr]

    node_ids = []
    for attr_v, subtree in tree.children.items():
        node_ids.append(f"{random.randint(0,100000)}")
        node_name = f'{f_name}: {value_names[attr_v]}'
        dot_str += f'    {curr} -> {node_ids[-1]} \n {node_ids[-1]} [label="{node_name} \\n {subtree.leaf_value}"]\n'

    i = 0
    for attr_v, subtree in tree.children.items():
        node_name = f'{f_name}: {value_names[attr_v]}'
        dot_str += f'{node_ids[i]} [label="{node_name} \\n {subtree.leaf_value}"] \n'
        dot_str += dot_string_rec(node_ids[i], subtree, feature_names = feature_names, feature_values = feature_values, class_names = class_names)
        i += 1

    return dot_str

if __name__ == '__main__':
    leaf_1 = DecisionTreeLeaf(1) # YES
    leaf_0 = DecisionTreeLeaf(0) # NO

    HUMEDAD, PRONOSTICO, VIENTO = 0, 1, 2
    ALTA, NORMAL = 0, 1
    SOL, NUBLADO, LLUVIOSO = 0, 1, 2
    FUERTE, DEBIL = 0, 1

    # Árbol de la diapositiva 28/65
    root = DecisionTree(PRONOSTICO, children = {
        SOL: DecisionTree(HUMEDAD, children = {ALTA: leaf_0, NORMAL: leaf_1}, best_class=None),
        NUBLADO: leaf_1,
        LLUVIOSO: DecisionTree(VIENTO, children = {FUERTE: leaf_0, DEBIL: leaf_1}, best_class=None),
    }, best_class=None)

    print(root.predict((1, 1, 1)))
    print(root.predict((0, 1, 0)))

