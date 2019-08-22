import numpy as np 

class BinaryNaiveBayesClassifier:
    """ Classifier for boolean attributes """

    def __init__(self, attrs):
        self.attrs = attrs      # Attribute names (order matters)

        self.classified = {}    # class -> [ attribute ]
        self.counts = {}        # class -> count
        self.priors = {}        # class -> [ attribute_frequency ]
        self.n_examples = 0     # size of the training set
        
        self.fitted = False
        

    def train(self, examples):
        """ Feed examples to the training set. 
            examples should be an iterable with k+1 elements,
            with 0/1 in the i-th position (0 < i < k) indicating
            False/True for the i-th attribute in this example.
            
            The last element of the example iterable should be its class
            (it can any hashable type)"""       
        self.fitted = False
        for case in examples:
            if(len(case) < len(self.attrs) + 1):
                raise f"Example {case} is malformed"

            klass = case[-1]
            attributes = np.array(case[:-1])

            if klass not in self.classified:
                self.classified[klass] = np.zeros(len(attributes))
                self.counts[klass] = 0
            
            self.classified[klass] += attributes
            self.counts[klass] += 1
        self.n_examples += len(examples)
        
    def fit(self):
        """Fit the model with the provided examples (i.e, estimate the prior distributions)"""
        must_correct = False

        for k,v in self.classified.items():
            # Optimistically try to compute non-corrected probabilities
            # If we find a zero-frequency we stop doing this and recompute with Laplace smoothing
            must_correct = 0 in v # Frequency 0 for attribute in this class
            if must_correct:
                break         

            self.priors[k] = v / self.counts[k]
            
            # Last element is the frequency of the class 
            self.priors[k] = np.append(self.priors[k], self.counts[k] / self.n_examples) 
        
        if must_correct:
            # Laplace smoothing
            for k,v in self.classified.items():
                n_classes = len(self.classified.keys())
                self.priors[k] = (v + 1) / (self.counts[k] + n_classes)

                # Last element is the frequency of the class 
                self.priors[k] = np.append(self.priors[k], self.counts[k] / self.n_examples)

        self.fitted = True
    
    def predict(self, attr):
        if not self.fitted:
            self.fit()
        map_prob = 0
        map_class = None
        for k,v in self.priors.items():
            prob = 1
            for i in range(len(attr)):
                if(attr[i] == 1):
                    prob *= v[i]
                else:
                    prob *= (1 - v[i])
            
            prob *= v[-1] # Last element is frequency of the class
            if prob > map_prob:
                map_prob = prob
                map_class = k
        
        return (map_class, map_prob)


attrs = ["scones", "cerveza", "whiskey", "avena", "futbol"]
examples = [
    [0,0,1,1,1,"I"],
    [1,0,1,1,0,"I"],
    [1,1,0,0,1,"I"],
    [0,1,0,0,0,"I"],
    [0,1,0,0,1,"I"],
    [1,1,1,1,0,"E"],
    [1,0,0,1,1,"E"],
    [1,1,0,0,1,"E"]
]

#nbc = BinaryNaiveBayesClassifier(attrs)
#nbc.train(examples)
#nbc.fit()
#print(nbc.predict([0,0,1,1,1]))

# El ejemplo del ppt 
# Tomo 'a' como verdadero y 'b' como falso
e2 = [
    [1, 0, 1, '+'],
    [0, 0, 0, '+'],
    [1, 1, 0, '-'],
    [0, 1, 0, '-']
]

nbc = BinaryNaiveBayesClassifier(['a1', 'a2', 'a3'])
nbc.train(e2)
nbc.fit()
print(nbc.predict([1,1,1]))
