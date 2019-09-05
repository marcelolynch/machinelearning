import csv
import os
import numpy as np
from collections import Counter
from multinomial_bayes import MultinomialBayesClassifier
from preprocessor import tokenize_document, basic_tokenize_document
from model_selector import KFold, Bootstrap
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
parser.add_argument('--dataset', choices=['aizemberg', 'reuters'], default='aizemberg')
parser.add_argument('--nsplits', action="store", type=int, default = 10)
parser.add_argument('--max_classes', action="store", type=int, default = -1)
parser.add_argument('--train_ratio', action="store", type=restricted_float, default = 1)
parser.add_argument('--random_seed', action="store", type=int, default = 1)
parser.add_argument('--confusion_matrix', action='store_true')
parser.add_argument('--normalize', action='store_true')

# Sample run
# python evaluate.py --validation bootstrap --nsplits 3 --train_ratio .2 --confusion_matrix

args = parser.parse_args()

def authors(max_classes = -1):
    texts = []

    authors = ["AaronPressman", "AlanCrosby", "AlexanderSmith", "BenjaminKangLim", "BernardHickey", "BradDorfman", "DarrenSchuettler", "DavidLawder", "EdnaFernandes", "EricAuchard", "FumikoFujisaki", "GrahamEarnshaw", "HeatherScoffield", "JaneMacartney", "JanLopatka", "JimGilchrist", "JoeOrtiz", "JohnMastrini", "JonathanBirt", "JoWinterbottom", "KarlPenhaul", "KeithWeir", "KevinDrawbaugh", "KevinMorrison", "KirstinRidley", "KouroshKarimkhany", "LydiaZajc", "LynneO'Donnell", "LynnleyBrowning", "MarcelMichelson", "MarkBendeich", "MartinWolk", "MatthewBunce", "MichaelConnor", "MureDickie", "NickLouth", "PatriciaCommins", "PeterHumphrey", "PierreTran", "RobinSidel", "RogerFillion", "SamuelPerry", "SarahDavison", "ScottHillis", "SimonCowell", "TanEeLyn", "TheresePoletti", "TimFarrand", "ToddNissen", "WilliamKazer"]
    for author in authors[0:max_classes]:
        for filename in os.listdir(f'editoriales/{author}'):
            with open(f'editoriales/{author}/{filename}') as textos:
                lines = [(line, author) for line in textos.readlines()]
                texts.extend(lines[:-1])
    return texts

def headlines():
    texts = []
    with open('aa_bayes.tsv', encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter = "\t")
        next(reader)   # Skip header
        i = 0
        for row in reader:
            if(len(row) > 3):
                texts.append((row[1], row[3]))

    texts = texts[:30000] # Remove Noticias Destacadas
    return texts

universe = set()
examples = []


texts = headlines() if args.dataset == "aizemberg" else authors(args.max_classes)
for text, klass in texts:
    histogram = basic_tokenize_document(text)
    universe.update(histogram.keys())
    examples.append((histogram, klass))

examples = np.array(examples)

# Label of each class
classes = list(set(c for t, c in texts))

model_selectors = { 
    'kfold': KFold(n_splits = args.nsplits if args.nsplits > 1 else 2, random_state = args.random_seed),
    # TODO: check if this is really bootstrap
    'bootstrap': Bootstrap(n_splits = args.nsplits, train_size = args.train_ratio, random_state = args.random_seed)
}

model_selector = model_selectors[args.validation]
print(f'Running {args.validation} validation')
print(args, '\n')

split = 1
for train_index, test_index in model_selector.split(examples):
    print(f"---- Split {split} ----\n")
    
    split += 1
    train = examples[train_index]
    test = examples[test_index]
    
    mbc = MultinomialBayesClassifier()
    mbc.fit(universe, train)
    score(mbc, test, classes, confusion_matrix = args.confusion_matrix, normalize = args.normalize)
    print()

