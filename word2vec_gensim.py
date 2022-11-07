from gensim.models import Word2Vec
from nltk.corpus import brown

import urllib
import re


def get_data():
    # change to your own path if you have downloaded the file locally
    url = 'https://dataskat.s3.eu-west-3.amazonaws.com/data/Shakespeare_alllines.txt'
    # read file into list of lines
    lines = urllib.request.urlopen(url).read().decode('utf-8').split("\n")

    sentences = []
    for line in lines:
        # remove punctuation
        line = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', '', line).strip()
        # tokenizer
        tokens = re.findall(r'\b\w+\b', line)
        if len(tokens) > 1:
            sentences.append(tokens)

    return sentences


if __name__ == '__main__':

    sentences = get_data()

    bard2vec = Word2Vec(
                sentences,
                min_count=3,   # Ignore words that appear less than this
                vector_size=50,# Dimensionality of word embeddings
                sg = 1,        # skipgrams
                window=7,      # Context window for words during training
                epochs=40)       # Number of epochs training over corpus

    print("Hello")