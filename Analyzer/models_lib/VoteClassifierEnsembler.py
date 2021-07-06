import os

import nltk
import joblib
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI


class EnsembleClassifier(ClassifierI):

    __word_features = None

    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self.__load_word_features()

    def __load_word_features(self):
        path = os.path.join(os.path.dirname(__file__), "vocabulary_model_objects", "word_features5k.save")
        self.__word_features = joblib.load(path)

    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # a simple measurement the degree of confidence in the classification
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for w in self.__word_features:
            features[w] = (w in words)

        return features

