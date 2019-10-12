#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import operator
import numpy as np
import string

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def cos(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

def get_candidates(lemma, pos):
    # Part 1
    synsets = wn.synsets(lemma, pos)
    lemmas = set()
    for synset in synsets:
        for l in synset.lemmas():
            ln = str(l.name())
            ln = ln.replace('_', ' ')
            lemmas.add(ln)
    if lemma in lemmas:
        lemmas.remove(lemma)
    return lemmas

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):

    synsets = wn.synsets(context.lemma, context.pos)
    lemmas = {}
    for synset in synsets:
        for l in synset.lemmas():
            ln = str(l.name())
            if context.lemma.lower() == ln.lower():
                continue
            ln = ln.replace('_', ' ')
            lemmas[ln] = l.count()

    # used this for getting max dict value
    # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

    return max(lemmas.items(), key=operator.itemgetter(1))[0] # replace for part 2

def wn_simple_lesk_predictor(context):
    best_sense = []
    max_overlap = 0

    synsets = wn.synsets(context.lemma, context.pos)
    context_candidates = get_candidates(context.lemma, context.pos)
    stop_words = stopwords.words('english')
    context_candidates = context_candidates.difference(stop_words)

    lc = set(context.left_context).difference(stop_words)
    rc = set(context.right_context).difference(stop_words)

    ctx = lc.union(rc)

    for synset in synsets:
        # making sure it's not self-referencing
        if synset.lemmas()[0].name().lower() == context.lemma.lower():
            continue

        tokens = tokenize(synset.definition())
        for ex in synset.examples():
            tokens = tokens + tokenize(ex)

        for hn in synset.hypernyms():
            tokens = tokens + tokenize(hn.definition())
            for ex in hn.examples():
                tokens = tokens + tokenize(ex)

        overlap = 0
        #compute overlap
        for t in tokens:
            if t in ctx and t not in stop_words:
                overlap += 1
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense.append(synset)

    if len(best_sense) == 0:
        return wn_frequency_predictor(context)
    elif len(best_sense) == 1:
        val = best_sense[0].lemmas()[0].name().lower()
        val = val.replace('_', ' ')
        return val
    else:
        val = ""
        count = 0
        for sense in best_sense:
            if sense.lemmas()[0].count() > count:
                count = sense.lemmas()[0].count()
                val = sense.lemmas()[0].name().lower()
        val = val.replace('_', ' ')
        return val
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        synonyms = get_candidates(context.lemma, context.pos)

        most_similar = ""
        max_compare = float("-inf")
        for syn in synonyms:
            syn = syn.replace(' ', '_')
            try:
                compare = self.model.similarity(context.lemma, syn)
                if compare > max_compare:
                    max_compare = compare
                    most_similar = syn
            except:
                continue

        return most_similar # replace for part 4

    def predict_nearest_with_context(self, context): 
        target_vector = self.model.wv[context.lemma]
        stop_words = stopwords.words('english')

        adjacent_words = {}
        adjacent_vector = None

        left_context = context.left_context[::-1]
        right_context = context.right_context

        count = 0
        for i in range(0, len(left_context)):
            if count > 4:
                break
            count += 1
            try:
                if left_context[i] not in stop_words:
                    adjacent_words[left_context[i]] = self.model.wv[left_context[i]]
            except:
                continue

        count = 0
        for i in range(0, len(right_context)):
            if count > 4:
                break
            if right_context[i] in string.punctuation:
                continue
            count += 1
            try:
                if right_context[i] not in stop_words:
                    adjacent_words[right_context[i]] = self.model.wv[right_context[i]]
            except:
                continue

        for word in adjacent_words:
            target_vector = np.add(target_vector, self.model.wv[word])

        synonyms = get_candidates(context.lemma, context.pos)

        most_similar = ""
        max_compare = float("-inf")

        for syn in synonyms:
            syn = syn.replace(' ', '_')
            try:
                compare = cos(self.model.wv[syn], target_vector)
                if compare > max_compare:
                    max_compare = compare
                    most_similar = syn
            except:
                continue

        return most_similar

    # combination of lesk (using definitions and examples) and cosine distance.
    # Takes the cosine distance of the definitions instead of just the context and synonyms
    # Hopefully this can get a closer replacement by comparing the similarity of their definitions
    # instead of just the similarity of the synonyms
    def best_predictor(self, context):
        target_vector = self.vectorize_token(context.lemma, context.pos)
        stop_words = stopwords.words('english')

        adjacent_words = {}
        adjacent_vector = None

        left_context = context.left_context[::-1]
        right_context = context.right_context
        context_vector = None

        count = 0
        for i in range(0, len(left_context)):
            if count > 4:
                break
            count += 1
            try:
                if left_context[i] not in stop_words:
                    if context_vector is None:
                        context_vector = self.vectorize_token(left_context[i])
                    else:
                        context_vector = np.add(self.vectorize_token(left_context[i]), context_vector)
            except:
                continue

        count = 0
        for i in range(0, len(right_context)):
            if count > 4:
                break
            if right_context[i] in string.punctuation:
                continue
            count += 1
            try:
                if right_context[i] not in stop_words:
                    if context_vector is None:
                        context_vector = self.vectorize_token(right_context[i])
                    else:
                        context_vector = np.add(self.vectorize_token(right_context[i]), context_vector)
            except:
                continue

        target_vector = np.add(context_vector, target_vector)
        synonyms = get_candidates(context.lemma, context.pos)

        most_similar = ""
        max_compare = float("-inf")

        for syn in synonyms:
            syn = syn.replace(' ', '_')
            try:
                replacement_vector = self.vectorize_token(syn, context.pos)
                compare = cos(replacement_vector, target_vector)
                if compare > max_compare:
                    max_compare = compare
                    most_similar = syn
            except:
                continue

        return most_similar

    # Creates a wordvector of a token using the definitions and examples
    def vectorize_token(self, word, pos=None):
        target_vector = None
        stop_words = stopwords.words('english')

        synset = None
        if pos is not None:
            synset = wn.synsets(word, context.pos)[0]
        else:
            synset = wn.synsets(word)[0]

        tokens = tokenize(synset.definition())
        for ex in synset.examples():
            tokens = tokens + tokenize(ex)

        for hypernym in synset.hypernyms():
            tokens = tokens + tokenize(hypernym.definition())
            for ex in hypernym.examples():
                tokens = tokens + tokenize(ex)

        for token in tokens:
            if token in stop_words:
                continue
            if target_vector is None:
                try:
                    target_vector = self.model.wv[token]
                except KeyError:
                    continue
            else:
                try:
                    target_vector = np.add(target_vector, self.model.wv[token])
                except KeyError:
                    continue

        return target_vector

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = wn_frequency_predictor(context) 
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest(context) 
        # prediction = predictor.predict_nearest_with_context(context) 
        prediction = predictor.best_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
