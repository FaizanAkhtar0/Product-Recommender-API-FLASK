from Analyzer.models_lib.sentiment_analyzer import SentimentAnalyzer


class ReviewAndRatingPreferenceModel:

    _keys = [1, 2, 3, 4, 5]
    _key_default_map = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    def __init__(self):
        self._analyzer = SentimentAnalyzer()
        self._df = None
        self._review_label = []
        self.progress_percentage = 0

    def _fill_missing_keys(self, rating_counts_per_identifier):
        present_keys = [k[0] for k in rating_counts_per_identifier]
        for k in self._keys:
            if k not in present_keys:
                rating_counts_per_identifier.append((k, self._key_default_map.get(k)))
        else:
            return rating_counts_per_identifier

    def map_rating_count_as_dict(self, rating_count_list):
        converted_dict = {}
        for item in rating_count_list:
            converted_dict[str(item[0])] = item[1]
        else:
            return converted_dict.copy()

    def extract_stats(self, df):
        whole_progress_counter = len(df)
        current_progress_counter = 0
        stats = []
        identifiers = df.identifier.unique()
        for identifier in identifiers:
            pos_count = 0
            neg_count = 0
            reviews_per_identifier = df.review[df.identifier == identifier]
            ratings_per_identifier = df.rating[df.identifier == identifier]
            for review in reviews_per_identifier:
                prediction, confidence = self._analyzer.Sentiment(str(review), nlp=False)
                if prediction == 'pos':
                    pos_count += 1
                else:
                    neg_count += 1
                current_progress_counter += 1
                self.progress_percentage = round((current_progress_counter / whole_progress_counter), 2)
                print("{:.3f}% - Complete".format(self.progress_percentage * 100))
            else:
                average_rating = (ratings_per_identifier.sum() / len(ratings_per_identifier))
                rating_counts_per_identifier = [(
                    ratings_per_identifier.value_counts().index[i], list(ratings_per_identifier.value_counts())[i]
                ) for i in range(len(ratings_per_identifier.value_counts()))]
                rating_counts_per_identifier = self._fill_missing_keys(rating_counts_per_identifier)
                rating_counts_per_identifier.sort(key=lambda x: x[0], reverse=True)
                rating_counts_per_identifier = self.map_rating_count_as_dict(rating_counts_per_identifier)
                total_rating_count = len(ratings_per_identifier)
                stats.append((
                    identifier, len(reviews_per_identifier), pos_count, neg_count, total_rating_count, round(average_rating, 1),
                    rating_counts_per_identifier
                ))
        else:
            return stats

    def recommendation(self, df):
        statistics = {}
        stats = self.extract_stats(df=df)
        stats.sort(key=lambda x: x[5], reverse=True)
        for stat in stats:
            identifier, total_reviews, pos_review_count, neg_review_count, total_rating_count, \
            avg_rating, rating_counts = stat
            history = {
                'total_reviews': total_reviews, 'positive_review_count': pos_review_count,
                'negative_review_count': neg_review_count, 'total_rating_count': total_rating_count,
                'average_rating': avg_rating, 'category_rating_counts': rating_counts
            }
            statistics[identifier] = history.copy()
        else:
            return statistics.copy()

