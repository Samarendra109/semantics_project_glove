import gensim
import logging

from gensim.models import Word2Vec
from nltk.corpus import brown

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = brown.sents()
model = Word2Vec(sentences, min_count=3, vector_size=25, sg = 1, window=10, epochs=100)
model.save('/tmp/brown_model')

#path = r"D:\UofT Courses\Computational Models of Semantic Change\Assignment 1\\"
path_analogy = "word-test.v1.txt"
#path_pair = path+"word_rg64_pairs.txt"

result_lsa_score, result_lsa_by_section = model.wv.evaluate_word_analogies(path_analogy)

print(result_lsa_score*100)

semantic_lsa_correct = 0
syntactic_lsa_correct = 0

semantic_lsa_total = 0
syntactic_lsa_total = 0

for d in result_lsa_by_section:
    if d['section'] == "Total accuracy":
        continue
    if d['section'].startswith('gram'):
        syntactic_lsa_correct += len(d['correct'])
        syntactic_lsa_total += len(d['correct']) + len(d['incorrect'])
    else:
        semantic_lsa_correct += len(d['correct'])
        semantic_lsa_total += len(d['correct']) + len(d['incorrect'])

semantic_lsa_score = semantic_lsa_correct / semantic_lsa_total
syntactic_lsa_score = syntactic_lsa_correct / syntactic_lsa_total

print((semantic_lsa_score*100,syntactic_lsa_score*100))


print("Hello")