import csv
import random
from DecisionTree import dot_string
from metrics import score, accuracies
import matplotlib.pyplot as plt
from graphviz import Source

def get_data(filename, separator, attributes):
    lines = []
    uniq_values = [set() for _ in attributes]
    with open(filename, encoding="utf8") as f:
        reader = csv.reader(f, delimiter = separator)
        next(reader)   # Skip header
        for row in reader:
            e = []
            skip = False
            for i,a in enumerate(attributes):
                v = a["transform"](row[a["idx"]])
                if v is None:
                    skip = True
                uniq_values[i].add(v)
                e.append(v)
            if not skip:
                lines.append(e)
    random.seed(2)
    random.shuffle(lines)   

    d = []
    for i,a in enumerate(attributes):
        d.append({"name": a['name'], "values": list(uniq_values[i])})
    return [x[0:-1] for x in lines], [x[-1] for x in lines], d


def get_graphviz(tree, d, options='rankdir="LR";'):
    feature_names = {}
    for i,x in enumerate(d[:-1]):
        feature_names[i] = x["name"]

    feature_values = {}
    for i,x in enumerate(d[:-1]):
        feature_values[i] = {}
        for a in x['values']:
            feature_values[i][a] = a

    class_names = {}
    for x in d[-1]["values"]:
        class_names[x] = x

    return Source(dot_string(tree, feature_names = feature_names, feature_values = feature_values, class_names = class_names, options=options))