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