import numpy as np 
from math import log, factorial


class MultinomialBayesClassifier:        
    def __init__(self):
        self.n_classes = 0
        self.distributions = {} # multinomial distributions for features in a class
        self.class_priors = {}  # probability of every class

        # For every class store the Laplace smoothing for probability 0 here,
        # so we can reuse it
        self.zero_smoothed = {} 
        
        self.fitted = False

    
    def fit(self, universe, examples):
        # examples :: [(feature -> freq , class)]
        self.universe = universe
        
        class_counts = {}

        class_histograms = {}
        for histogram, klass in examples:
            if klass not in class_histograms:
                class_histograms[klass] = {}

            for word, freq in histogram.items():
                if word not in universe:
                    # Only consider features in the universe
                    continue

                if word not in class_histograms[klass]:
                    class_histograms[klass][word] = 0
                    class_counts[klass] = 0

                class_histograms[klass][word] += freq
                class_counts[klass] += freq
        
        # With the consolidated histograms we build the distributions
        n_classes = len(class_histograms.keys())
        for klass, histogram in class_histograms.items():
            if klass not in self.distributions:
                self.distributions[klass] = {}
            for word, freq in histogram.items():
                # Always apply Laplace smoothing
                self.distributions[klass][word] = (freq + 1)/(class_counts[klass] + n_classes)
        
        for klass in class_histograms.keys():
            self.zero_smoothed[klass] = 1/(class_counts[klass] + n_classes)
            self.class_priors[klass] = class_counts[klass] / len(examples)

        self.fitted = True

    def predict(self, observed):
        """ Classify the observed histogram in a known class.
            
            Parameters
            ----------
            observed : 
                dict(_, int). The histogram for the target case  
        """
        if not self.fitted:
            raise "Must fit the model before predicting"

        map_log_prob = -100000000
        map_class = None

        for klass, distribution in self.distributions.items():
            # Use log-space for better stability
            log_prob = log(self.class_priors[klass])
            for feature, freq in observed.items():
                # If the key isn't present it means the probability is zero,
                # we get it from the previously calculated zero-probability for the class
                p = distribution.get(feature, self.zero_smoothed[klass])
                log_prob += freq * log(p)
            if log_prob > map_log_prob:
                map_log_prob = log_prob
                map_class = klass
        
        return map_class         


# ============= Example ====================
import csv
from collections import Counter

texts = []
with open('textminingAllLyrics.csv') as tsvfile:
  reader = csv.reader(tsvfile)
  next(reader)
  i = 0
  for row in reader:
    texts.append((row[1], row[2]))
      
universe = set()
examples = []
for text, klass in texts:
    tarr = text.split(' ')
    universe.update(tarr)
    examples.append((Counter(tarr), klass))

train = []
evaluate = []
for i in range(len(examples)):
    if i % 40 == 0:
        evaluate.append(examples[i])
    else:
        train.append(examples[i])

mbc = MultinomialBayesClassifier()
mbc.fit(universe, train)

correct = 0
total = 0
for ex in evaluate:
    if (ex[1] == mbc.predict(ex[0])):
        correct += 1
    total += 1

print(correct/total)
