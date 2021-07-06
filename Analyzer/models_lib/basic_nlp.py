import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


class NLP:
    is_sentence = False

    def tokenize(self, document, sent_token=False):
        self.is_sentence = sent_token
        if sent_token:
            return [sent for sent in sent_tokenize(document)]
        else:
            return [word for word in word_tokenize(document)]

    def remove_special_characters(self, token_text, remove_digits=False):
        filtered_token_text = []
        for item in token_text:
            pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
            text = re.sub(pattern, '', item)
            if text != '':
                filtered_token_text.append(text)

        return filtered_token_text

    def make_lower(self, document):
        return [item.lower() for item in document]

    def remove_stopwords(self, document):

        is_sent = self.is_sentence
        stop_words = stopwords.words('english')

        if is_sent:
            filtered_sentences = []
            for sentence in document:
                temp_filtered_sentence = ' '.join([word for word in word_tokenize(sentence) if word not in stop_words])
                filtered_sentences.append(temp_filtered_sentence)
            return filtered_sentences
        else:
            return [word for word in document if word not in stop_words]

    def __get_base_mapping_pos(self, pen_tag):
        """Returns base mapping for a tag as each tag belongs to a specific base class"""
        base_mapping = {
            'NN': 'n', 'JJ': 'a',
            'VB': 'v', 'RB': 'r'
        }

        try:
            return base_mapping[pen_tag[:2]]
        except:
            return 'n'

    def lemmatize(self, document):

        is_sent = self.is_sentence
        w_n_lemmatizer = WordNetLemmatizer()

        if is_sent:
            lemmatized_sentences = []
            for sentence in document:
                temp_sent = ' '.join([w_n_lemmatizer.lemmatize(word, pos=self.__get_base_mapping_pos(tag))
                                      for word, tag in pos_tag(word_tokenize(sentence))])
                lemmatized_sentences.append(temp_sent)
            return lemmatized_sentences
        else:
            return [w_n_lemmatizer.lemmatize(word, pos=self.__get_base_mapping_pos(pos_tag(word))) for word in document]


