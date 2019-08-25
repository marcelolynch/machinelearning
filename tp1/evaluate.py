import csv
import numpy as np
from collections import Counter
from multinomial_bayes import MultinomialBayesClassifier
from preprocessor import tokenize_document, basic_tokenize_document
from sklearn.model_selection import KFold, ShuffleSplit
from metrics import score
import matplotlib.pyplot as plt
import argparse

def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range (0.0, 1.0)")
    return x
    
parser = argparse.ArgumentParser(description='Evaluate Multinomial Bayes Classifier model.')
parser.add_argument('--validation', choices=['kfold', 'bootstrap'], default='kfold')
parser.add_argument('--nsplits', action="store", type=int, default = 10)
parser.add_argument('--test_ratio', action="store", dest="test_ratio", type=restricted_float, default = 0.1)
parser.add_argument('--random_seed', action="store", type=int, default = 1)
parser.add_argument('--confusion_matrix', action='store_true')

# Sample run
# python evaluate.py --validation bootstrap --nsplits 3 --test_ratio .2 --confusion_matrix

args = parser.parse_args()

texts = []
with open('aa_bayes.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter = "\t")
    next(reader)   # Skip header
    i = 0
    for row in reader:
        if(len(row) > 3):
            texts.append((row[1], row[3]))
    
universe = set()
examples = []
for text, klass in texts[:30000]:
    histogram = basic_tokenize_document(text)
    universe.update(histogram.keys())
    examples.append((histogram, klass))

examples = np.array(examples)

# Label of each class
classes = list(set(c for t, c in texts))

model_selectors = { 
    'kfold': KFold(n_splits = args.nsplits, random_state = args.random_seed),
    # TODO: check if this is really bootstrap
    'bootstrap': ShuffleSplit(n_splits = args.nsplits, test_size = args.test_ratio, random_state = args.random_seed)
}

model_selector = model_selectors[args.validation]
print(f'Running {args.validation} validation')
print(args, '\n')

split = 1
scores = []
for train_index, test_index in model_selector.split(examples):
    print(f"---- Split {split} ----")
    
    split += 1
    train = examples[train_index]
    test = examples[test_index]
    
    mbc = MultinomialBayesClassifier()
    mbc.fit(universe, train)
    s = score(mbc, test, classes, confusion_matrix = args.confusion_matrix)
    scores.append(s)
    print()

print(f'Average score: {np.mean(scores)}')