from random import Random

class KFold:
    def __init__(self, n_splits, random_state = 0):
        self._random = Random(random_state)
        self._n_splits = n_splits
    
    def split(self, examples):
        indices = list(range(len(examples)))
        self._random.shuffle(indices)
        test_size = len(examples)//self._n_splits
        for i in range(self._n_splits):
            test_begin = i*test_size
            test_end = test_begin + test_size + 1
            selected = indices[0:test_begin] + indices[test_end:-1]
            test = indices[test_begin:test_end]
            yield selected, test

    

class Bootstrap:
    def __init__(self, n_splits, train_size = 1, random_state = 0):
        self._random = Random(random_state)
        self._n_splits = n_splits
        self._train_size = train_size
    
    def split(self, examples): 
        indices = list(range(len(examples)))
        train_size = self._train_size * len(examples)
        for i in range(self._n_splits):
            train = self._random.choices(indices, k=train_size) # Draw with replacement
            used = set(train)
            test = [i for i in indices if i not in used]
            yield train, test

    






