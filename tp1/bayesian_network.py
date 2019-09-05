import itertools, csv

class DiscreteBayesianNetwork:
    def __init__(self, variables, parents_of):
        self.variables = variables # name -> possible_values
        self.parents_of = parents_of
        self.nodes = parents_of.keys()
        self.parent_conditionals = {}
    
    def _all_parent_combinations(self, node):
        """ Returns the cartesian product of the random variables that are parents of a node """ 
        if len(self.parents_of[node]) == 0:
            return [()]
        variable_values = [self.variables[p] for p in self.parents_of[node]]
        return itertools.product(*variable_values)

    def _set_probabilities(self, parent_probabilities):
        self.parent_conditionals = parent_probabilities
    
    def fit(self, examples):
        # examples :: [ node->value ]
        # Initialize the counters
        counts = {}
        conditional_counts = {}
        for node in self.nodes:
            counts[node] = {}
            conditional_counts[node] = {}
            for t in self._all_parent_combinations(node):
                conditional_counts[node][t] = {}     
                for v in self.variables[node]:
                    conditional_counts[node][t][v] = 0  # Count(node = v | parents = t)
                counts[node][t] = 0
  
        # Count all the conditional ocurrences in the examples
        for example in examples:
            for node, parents in self.parents_of.items():
                t = tuple([example[p] for p in parents])
                val = example[node]
                conditional_counts[node][t][val] += 1
                counts[node][t] += 1
        
        
        # Calculate probabilities from relative frequencies
        for node in self.nodes:
            self.parent_conditionals[node] = {}
            for t in self._all_parent_combinations(node):
                self.parent_conditionals[node][t] = {}
                for v in self.variables[node]:
                    self.parent_conditionals[node][t][v] = conditional_counts[node][t][v] / counts[node][t]      # P(node = v | parents = t)
    
    def _calculate_full_joint(self, targets):
        false_nodes = [x for x in targets.keys() if x not in self.nodes]
        if len([x for x in targets.keys() if x not in self.nodes]): # Key error, no such node
            raise Exception(f"Some targets are not nodes of this network: {false_nodes}")

        prob = 1
        for n, ps in self.parents_of.items():
            v = targets[n]
            t = tuple([targets[p] for p in ps])
            prob *= self.parent_conditionals[n][t][v]
        return prob
    
    def _calculate_joint(self, targets):
        absent_variables = [x for x in self.variables.keys() if x not in targets.keys()] # Variables missing from the 'full' joint probability
        
        if len(absent_variables) == 0:  # We have information for all variables: we don't need to use total probability
            return self._calculate_full_joint(targets)

        # Use the law of total probability        
        prob = 0
        absent_values = itertools.product(*[self.variables[x] for x in absent_variables]) # Partition 
        for value in absent_values:
            absent_dict = dict(zip(absent_variables, value))  # Turn tuple into a dict
            full_joint = {**targets, **absent_dict } # Merge dicts
            prob += self._calculate_full_joint(full_joint)
        return prob
    
    def calculate(self, targets, conditions = {}):
        if (len(conditions) == 0):
            return self._calculate_joint(targets)
        
        # P(A|B) = P(A,B)/P(B)
        intersection = {**targets, **conditions}
        return self._calculate_joint(intersection) / self._calculate_joint(conditions)


def ejercicio4():
    examples = []
    with open('binary.csv') as binary:
        reader = csv.reader(binary)
        next(reader)   # Skip header
        #admit,gre,gpa,rank
        for row in reader:
            fact = {}
            fact["admit"] = int(row[0])
            fact["gre"] = 0 if float(row[1]) >= 500 else 1
            fact["gpa"] = 0 if float(row[2]) >= 3 else 1
            fact["rank"] = int(row[3])
            examples.append(fact)

    
    dbn = DiscreteBayesianNetwork( {"admit": [0,1], "gre": [0,1], "gpa": [0,1], "rank": [1,2,3,4]},
                                    {
                                        "admit": ["rank", "gre", "gpa"],
                                        "gre": ["rank"],
                                        "gpa": ["rank"],
                                        "rank": []
                                    })
    dbn.fit(examples)

    # Ejercicio 4a
    print("Ejercicio 4a:")
    print(dbn.calculate( { "admit": 0 }, conditions = { "rank": 1 }))
    print()

    # Ejercicio 4b
    print("Ejercicio 4b:")
    print(dbn.calculate( { "admit": 1 }, conditions = { "rank": 2, "gre": 1, "gpa": 0 }))


def ejercicio1():
    dbn = DiscreteBayesianNetwork( {"Edad": ["V","J"], 
                                    "G1": ["Si", "No"], 
                                    "G2": ["Si", "No"], 
                                    "G3": ["Si", "No"], 
                                    "G4": ["Si", "No"]},
                                    {
                                        "Edad": [],
                                        "G1": ["Edad"],
                                        "G2": ["Edad"],
                                        "G3": ["Edad"],
                                        "G4": ["Edad"]
                                    })
    
    conditional_probabilities = {
        "Edad": { (): { "V": 0.9, "J": 0.1 } },
        "G1": {
            ("J",): { "Si": 0.95, "No": 0.05  },
            ("V",): { "Si": 0.03, "No": 0.97  }
        },
        "G2": {
            ("J",): { "Si": 0.05, "No": 0.95  },
            ("V",): { "Si": 0.82, "No": 0.18  }
        },
        "G3": {
            ("J",): { "Si": 0.02, "No": 0.98  },
            ("V",): { "Si": 0.34, "No": 0.66  }
        },
        "G4": {
            ("J",): { "Si": 0.2, "No": 0.8  },
            ("V",): { "Si": 0.92, "No": 0.08  }
        }
    }
    dbn._set_probabilities(conditional_probabilities)

    # Ejercicio 4a
    print("Probabilidad de ser joven dadas las preferencias:")
    ej1a = dbn.calculate( { "Edad": "J" }, conditions = { "G1": "Si", "G2": "No", "G3": "Si", "G4":"No" })
    print(ej1a)
    print()
    # Ejercicio 4b
    print("Probabilidad de ser viejo dadas las preferencias:")
    ej1b = dbn.calculate( { "Edad": "V" }, conditions = { "G1": "Si", "G2": "No", "G3": "Si", "G4":"No" })
    print(ej1b)

print("========== EJERCICIO 1 =============")
ejercicio1()
print("\n======== EJERCICIO 4 =============")
ejercicio4()