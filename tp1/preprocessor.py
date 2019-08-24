from nltk import ngrams, everygrams, word_tokenize
from collections import Counter
# from nltk.corpus import stopwords
from string import punctuation

""" Get the historgram of tokens for the given document
    after preprocessing.
"""
def tokenize_document(document, *, grams_count = 1):
    print(document)
    document = document.lower()

    # Remove punctuation
    document = document.translate(str.maketrans('', '', punctuation))

    # Get N_grams
    n_grams = everygrams(document.split(' '), 1, grams_count)
    n_grams_str = [' '.join(filter(None, g)) for g in n_grams]

    return Counter(n_grams_str)

# Dios no puedo creer que este funcione mejor que metiendo todo el 'preprocesamiento'
# O algo esta mal o es todo una cagada
def basic_tokenize_document(document, **kwargs):
    return Counter(document.split(' '))

        