import nltk
import torch
from nltk.corpus import brown
from nltk.util import ngrams
from collections import Counter, OrderedDict
import numpy as np

from nltk.corpus import stopwords
from string import punctuation

from tqdm import tqdm
from os.path import exists
import pickle

nltk.download('brown')
punctuation = set(list(punctuation))
context_size = 10

data_file = "training_data.pkl"
word2index_file = "word2index.pkl"


# def get_data():
#
#     if not (exists(data_file) and exists(word2index_file)):
#
#         words = brown.words()
#         words = [word.lower() for word in words]
#         words = [word for word in words if word.isalpha()]
#         words = [word for word in words if word not in punctuation]
#
#         f_words = nltk.FreqDist(words)
#         print(len(f_words))
#
#         most_common = sorted(f_words.most_common(5000), key=lambda tup: tup[1])
#         most_common_words = [tup[0] for tup in most_common]
#
#         final_words = set(most_common_words)
#         word2index = {w: i for i, w in enumerate(final_words)}
#
#         indices = OrderedDict()
#         values = []
#
#         context_windows_list = [tuple(a) for a in ngrams(words, context_size + 1)]
#         print(len(context_windows_list))
#
#         for context_window in tqdm(context_windows_list):
#
#             center_index = context_size // 2
#             center_word = context_window[center_index]
#
#             if center_word not in word2index:
#                 continue
#
#             for j, context_word in enumerate(context_window):
#
#                 if (context_word in word2index) and (j != center_index):
#                     index_of_words = (word2index[center_word], word2index[context_word])
#                     if index_of_words not in indices:
#                         indices[index_of_words] = len(values)
#                         values.append(0)
#                     values[indices[index_of_words]] += 1
#
#         torch_indices = torch.LongTensor(list(indices.keys()))
#         torch_values = torch.tensor(values)
#
#         with open(data_file, "wb") as f:
#             pickle.dump((torch_indices, torch_values), f)
#
#         with open(word2index_file, "wb") as f:
#             pickle.dump(word2index, f)
#
#     else:
#         with open(data_file, "rb") as f:
#             torch_indices, torch_values = pickle.load(f)
#
#     print("Entries", torch_values.size())
#     return torch_indices, torch_values

def get_context_windows():

    sentences = brown.sents()

    context_window_list = []
    for sent in tqdm(sentences):
        sent = [word.lower() for word in sent if (word.isalpha()) and (word not in punctuation)]
        for i, word in enumerate(sent[:-1]):
            context_window_list.append(tuple(sent[i:min(len(sent), i+context_size+1)]))

    return context_window_list

def get_data():

    if not (exists(data_file) and exists(word2index_file)):

        words = brown.words()
        words = [word.lower() for word in words]
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if word not in punctuation]

        f_words = nltk.FreqDist(words)
        print(len(f_words))

        most_common = sorted(f_words.most_common(5000), key=lambda tup: tup[1])
        most_common_words = [tup[0] for tup in most_common]

        final_words = set(most_common_words)
        word2index = {w: i for i, w in enumerate(final_words)}

        indices = OrderedDict()
        values = []

        context_windows_list = get_context_windows()
        print(len(context_windows_list))

        for context_window in tqdm(context_windows_list):

            center_word = context_window[0]

            if center_word not in word2index:
                continue

            for j, context_word in enumerate(context_window):

                if (context_word in word2index) and (context_word != center_word):
                    index_of_words = (word2index[center_word], word2index[context_word])
                    if index_of_words not in indices:
                        indices[index_of_words] = len(values)
                        values.append(0)
                    values[indices[index_of_words]] += 1/j

        torch_indices = torch.LongTensor(list(indices.keys()))
        torch_values = torch.tensor(values)

        with open(data_file, "wb") as f:
            pickle.dump((torch_indices, torch_values), f)

        with open(word2index_file, "wb") as f:
            pickle.dump(word2index, f)

    else:
        with open(data_file, "rb") as f:
            torch_indices, torch_values = pickle.load(f)

    print("Entries", torch_values.size())
    return torch_indices, torch_values


if __name__ == "__main__":
    get_data()
