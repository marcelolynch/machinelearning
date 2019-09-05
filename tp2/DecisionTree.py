class DecisionTree():
    def __init__(self, attr, *, children = {}):
        self.attr = attr
        self.children = children # Dict attr_value -> DecisionTree

    def predict(self, x):
        subtree = self.children[x[self.attr]]
        return subtree.predict(x)

    def is_leaf(self):
        return False

class DecisionTreeLeaf(DecisionTree):
    def __init__(self, leaf_value):
        self.leaf_value = leaf_value

    def predict(self, x):
        return self.leaf_value

    def is_leaf(self):
        return True

def dot_string(tree, *, feature_names, class_names):
    return 'digraph G {\n'  + dot_string_rec('root', tree, feature_names = feature_names, class_names = class_names) + '}'

def dot_string_rec(curr, tree, *, feature_names, class_names):
    dot_str = ''

    if tree.is_leaf():
        klass = class_names[tree.leaf_value]
        c = curr.replace('"', '')
        node_name = f'"{c}+{klass}"'
        return f'    {curr} -> {node_name}\n    {node_name} [label="{klass}"]\n'

    f_name = feature_names[tree.attr]
    for attr_v, subtree in tree.children.items():
        node_name = f'"{f_name}: {attr_v}"'
        dot_str += f'    {curr} -> {node_name}\n'

    for attr_v, subtree in tree.children.items():
        node_name = f'"{f_name}: {attr_v}"'
        dot_str += dot_string_rec(node_name, subtree, feature_names = feature_names, class_names = class_names)

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
        SOL: DecisionTree(HUMEDAD, children = {ALTA: leaf_0, NORMAL: leaf_1}),
        NUBLADO: leaf_1,
        LLUVIOSO: DecisionTree(VIENTO, children = {FUERTE: leaf_0, DEBIL: leaf_1}),
    })

    print(root.predict((1, 1, 1)))
    print(root.predict((0, 1, 0)))

