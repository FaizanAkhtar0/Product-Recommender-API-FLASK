import joblib
import sklearn
from Analyzer.models_lib.pre_trained import paths
from Analyzer.models_lib.VoteClassifierEnsembler import EnsembleClassifier
from Analyzer.models_lib.basic_nlp import NLP


class SentimentAnalyzer:

    _models = []

    def __init__(self):
        self.load_models()
        self._ensemble_vote_clf = EnsembleClassifier(*self._models)
        self._nlp = NLP()

    def load_models(self):
        for path in paths.return_path():
            self._models.append(joblib.load(path))

    def Sentiment(self, input_text, nlp=False):
        if nlp:
            filtered_text = ' '.join(self._nlp.lemmatize(self._nlp.make_lower(
                self._nlp.remove_stopwords(self._nlp.remove_special_characters(
                    self._nlp.tokenize(input_text, sent_token=True), remove_digits=True)))))
        else:
            filtered_text = input_text

        features = self._ensemble_vote_clf.find_features(filtered_text)
        return self._ensemble_vote_clf.classify(features), self._ensemble_vote_clf.confidence(features)



