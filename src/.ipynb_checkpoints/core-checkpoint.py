from lmc.Representations import *
from lmc.exps import *
import toml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# class QueryLearner:
#     def __init__(self):
#         self.learned_queries = {}
#         self.metadata = {}
#         self.prototype = None
#
#         self.encoding = None
#         self._load_encoding_from_disk()
#         self.onehot = None
#         self.positive_samples = None
#         self.negative_samples = None
#         self.likelihood_all_songs = None
#
#     def make_prototype(self, positive_samples):
#         onehot_positive = self.onehot.merge(pd.DataFrame(positive_samples, columns=['song_id']), on='song_id',
#                                             how='right')
#         onehot_positive = onehot_positive.drop(columns=['song_id', 'title', 'group'])
#         counts = np.array(onehot_positive.sum())
#         prototype = counts / onehot_positive.shape[0]
#         del onehot_positive
#         return prototype
#
#     def _load_encoding_from_disk(self):
#         with open(ENCODING_FP) as f:
#             e = toml.load(f)
#
#         for k, v in e.items():
#             e[k] = {int(a): b for a, b in v.items()}
#         self.encoding = e
#
#     def fit(self, positive_samples=None, discard_threshold=0.1, onehot=None, reliable_negative_method='l',
#             add_positives=False, strong_positive_threshold=0.3):
#         if positive_samples is None or onehot is None:
#             raise ValueError('invalid input!')
#
#         self.onehot = onehot
#         self.make_most_general_query(positive_samples)
#         self.prototype = self.make_prototype(positive_samples)
#         self.positive_samples = positive_samples
#
#         if reliable_negative_method == 'l':
#             self.negative_samples, added_positive_samples = self.get_new_samples_likelihood(positives=add_positives)
#         elif reliable_negative_method == 'r':
#             self.negative_samples, added_positive_samples = self.get_new_samples_rocchio(positives=add_positives)
#         elif reliable_negative_method == '1dnf':
#             self.negative_samples, added_positive_samples = \
#                 self.get_new_samples_1dnf(positives=add_positives, strong_positive_threshold=strong_positive_threshold)
#
#         self.positive_samples = list(set(self.positive_samples).union(added_positive_samples))
#
#         print('number of positive samples is {}, and number of negative samples is {}'.
#               format(len(self.positive_samples), len(self.negative_samples)))
#
#         tree_object = DTRepr(self.positive_samples, self.negative_samples, self.encoding, discard_threshold, onehot)
#
#         self.metadata['tree'] = tree_object
#         self.learned_queries['dt'] = tree_object.query_r
#         self.learned_queries['songs'] = tree_object.query_s
#         del tree_object
#
#     def make_most_general_query(self, positive_samples):
#         onehot_positive = self.onehot.merge(pd.DataFrame(positive_samples, columns=['song_id']), on='song_id',
#                                             how='right')
#         onehot_positive = onehot_positive.drop(columns=['song_id', 'title', 'group'])
#         representation = list(onehot_positive.sum())
#         representation = [v / onehot_positive.shape[0] for v in representation]
#         q = DTRepr.get_query_songs(representation, onehot_positive.columns, self.encoding)
#         self.learned_queries['most-general'] = q
#         del onehot_positive, representation
#
#     def get_new_samples_likelihood(self, negatives=True, positives=False):
#         likelihood_all_songs = self.get_likelihood_all_songs()
#         self.likelihood_all_songs = likelihood_all_songs
#         negative_samples = {}
#         if negatives:
#             negative_samples = likelihood_all_songs[likelihood_all_songs['likelihood'] == 0]['song_id'].tolist()
#
#         new_positive_samples = {}
#         if positives:
#             new_positive_samples = likelihood_all_songs.query('likelihood > likelihood.quantile(.99)').song_id.tolist()
#
#         del likelihood_all_songs
#         return negative_samples, new_positive_samples
#
#     def get_new_samples_rocchio(self, negatives=True, positives=False):
#         assert (not positives)
#         positive_prototype = self.prototype
#         unlabelled_samples = list(set(self.onehot.song_id) - set(self.positive_samples))
#         unlabelled_prototype = self.make_prototype(unlabelled_samples)
#         negative_samples = []
#
#         def compare_distance(l):
#             values = l.values
#             if np.linalg.norm(values[3:] - positive_prototype) > np.linalg.norm(values[3:] - unlabelled_prototype):
#                 negative_samples.append(values[0])
#
#         if negatives:
#             self.onehot[self.onehot['song_id'].isin(unlabelled_samples)].apply(compare_distance, axis=1)
#         return negative_samples, {}
#
#     def get_new_samples_1dnf(self, negatives=True, positives=False, strong_positive_threshold=0.3):
#         assert (not positives)
#         negative_samples = []
#
#         unlabelled_samples = list(set(self.onehot.song_id) - set(self.positive_samples))
#         strong_positive_features_indices = [i for i in range(len(self.prototype)) if
#                                             self.prototype[i] >= strong_positive_threshold]
#
#         def filter_data(l):
#             values = l.values
#             if all([values[3:][i] == 0 for i in strong_positive_features_indices]):
#                 negative_samples.append(values[0])
#
#         if negatives:
#             self.onehot[self.onehot['song_id'].isin(unlabelled_samples)].apply(filter_data, axis=1)
#         return negative_samples, {}
#
#     def get_likelihood_all_songs(self):
#         likelihood_all_songs = pd.DataFrame()
#         indices = [0] + [len(self.encoding[k]) for k in self.encoding]
#         indices = [sum(indices[:i]) for i in range(1, len(indices) + 1)]
#         likelihoods = np.array([1] * self.onehot.shape[0])
#         onehot_prototype = self.onehot.iloc[:, 3:].mul(self.prototype)
#
#         for k in range(1, len(indices)):
#             likelihoods = likelihoods * np.array(
#                 onehot_prototype.iloc[:, indices[k - 1]:indices[k]].sum(axis=1))
#
#         likelihood_all_songs['song_id'] = self.onehot.iloc[:, 0]
#         likelihood_all_songs['likelihood'] = likelihoods
#         likelihood_all_songs = likelihood_all_songs.sort_values(by=['likelihood'], ascending=False)
#         likelihood_all_songs.reset_index(inplace=True, drop=True)
#         del indices, onehot_prototype
#         return likelihood_all_songs
#
#     def score(self, all_context_songs):
#         return {'most-general': self._score_most_general_query(all_context_songs),
#                 'songs-query': self._score_songs_query(all_context_songs),
#                 'dt-query': self._score_dt_query(all_context_songs)}
#
#     def _score_dt_query(self, all_context_songs):
#         fetched_songs = QueryLearner.get_songs_from_query(self.learned_queries['dt'], query_type='dt')
#         return evaluator(predicted=fetched_songs, target=all_context_songs)
#
#     def _score_songs_query(self, all_context_songs):
#         fetched_songs = QueryLearner.get_songs_from_query(self.learned_queries['songs'], query_type='songs')
#         return evaluator(predicted=fetched_songs, target=all_context_songs)
#
#     def _score_most_general_query(self, all_context_songs):
#         fetched_songs = QueryLearner.get_songs_from_query({'most-general': self.learned_queries['most-general']},
#                                                           query_type='songs')
#         return evaluator(predicted=fetched_songs, target=all_context_songs)
#
#     @staticmethod
#     def get_songs_from_query(query, query_type=None):
#         if query_type == 'dt':
#             return get_songs_from_query_dt_rules(query)
#         elif query_type == 'songs':
#             return get_songs_from_query_song_query(query)
#         else:
#             raise ValueError('invalid query type: {}'.format(query_type))


class QueryLearner:
    def __init__(self):
        self.learned_queries = {}
        self.metadata = {}
        self.prototype = None
        self.unlabelled_prototype = None

        self.encoding = None
        self._load_encoding_from_disk()
        self.onehot = None
        self.positive_samples = None
        self.negative_samples = None
        self.likelihood_all_songs = None

    def make_prototype(self, positive_samples):
        onehot_positive = self.onehot.merge(pd.DataFrame(positive_samples, columns=['song_id']), on='song_id',
                                            how='right')
        onehot_positive = onehot_positive.drop(columns=['song_id', 'title', 'group'])
        counts = np.array(onehot_positive.sum())
        prototype = counts / onehot_positive.shape[0]
        del onehot_positive
        return prototype

    def _load_encoding_from_disk(self):
        with open(ENCODING_FP) as f:
            e = toml.load(f)

        for k, v in e.items():
            e[k] = {int(a): b for a, b in v.items()}
        self.encoding = e

    def fit(self, positive_samples=None, negative_samples=None, discard_threshold=0.1, onehot=None,
            reliable_negative_method='l', add_positives=False, strong_positive_threshold=0.3, rocchio_threshold=0):
        if positive_samples is None or onehot is None:
            raise ValueError('invalid input!')

        self.onehot = onehot
        self.make_most_general_query(positive_samples)
        self.prototype = self.make_prototype(positive_samples)
        self.positive_samples = positive_samples

        if negative_samples is None:
            added_positive_samples = {}
            if reliable_negative_method == 'l':
                self.negative_samples, added_positive_samples = self.get_new_samples_likelihood(positives=add_positives)
            elif reliable_negative_method == 'r':
                self.negative_samples, added_positive_samples = \
                    self.get_new_samples_rocchio(positives=add_positives, rocchio_threshold=rocchio_threshold)
            elif reliable_negative_method == '1dnf':
                self.negative_samples, added_positive_samples = \
                    self.get_new_samples_1dnf(positives=add_positives,
                                              strong_positive_threshold=strong_positive_threshold)

            self.positive_samples = list(set(self.positive_samples).union(added_positive_samples))
        else:
            self.negative_samples = negative_samples

        print('number of positive samples is {}, and number of negative samples is {}'.
              format(len(self.positive_samples), len(self.negative_samples)))

        tree_object = DTRepr(self.positive_samples, self.negative_samples, self.encoding, discard_threshold, onehot)

        self.metadata['tree'] = tree_object
        self.learned_queries['dt'] = tree_object.query_r
        self.learned_queries['songs'] = tree_object.query_s
        del tree_object

    def make_most_general_query(self, positive_samples):
        onehot_positive = self.onehot.merge(pd.DataFrame(positive_samples, columns=['song_id']), on='song_id',
                                            how='right')
        onehot_positive = onehot_positive.drop(columns=['song_id', 'title', 'group'])
        representation = list(onehot_positive.sum())
        representation = [v / onehot_positive.shape[0] for v in representation]
        q = DTRepr.get_query_songs(representation, onehot_positive.columns, self.encoding)
        self.learned_queries['most-general'] = q
        del onehot_positive, representation

    def get_new_samples_likelihood(self, negatives=True, positives=False):
        likelihood_all_songs = self.get_likelihood_all_songs()
        self.likelihood_all_songs = likelihood_all_songs
        negative_samples = {}
        if negatives:
            negative_samples = likelihood_all_songs[likelihood_all_songs['likelihood'] == 0]['song_id'].tolist()

        new_positive_samples = {}
        if positives:
            new_positive_samples = likelihood_all_songs.query('likelihood > likelihood.quantile(.99)').song_id.tolist()

        del likelihood_all_songs
        return negative_samples, new_positive_samples

    # def get_new_samples_rocchio_1(self, negatives=True, positives=False):
    #     assert (not positives)
    #     positive_prototype = self.prototype
    #     unlabelled_samples = list(set(self.onehot.song_id) - set(self.positive_samples))
    #     unlabelled_prototype = self.make_prototype(unlabelled_samples)
    #     self.unlabelled_prototype = unlabelled_prototype
    #     negative_samples = []
    #
    #     def compare_distance(l):
    #         values = l.values
    #         if np.linalg.norm(values[3:] - positive_prototype) > np.linalg.norm(values[3:] - unlabelled_prototype):
    #             negative_samples.append(values[0])
    #
    #     if negatives:
    #         self.onehot[self.onehot['song_id'].isin(unlabelled_samples)].apply(compare_distance, axis=1)
    #     return negative_samples, {}

    def get_new_samples_rocchio(self, negatives=True, positives=False, rocchio_threshold=0):
        assert (not positives)
        positive_prototype = self.prototype
        unlabelled_samples = list(set(self.onehot.song_id) - set(self.positive_samples))
        unlabelled_prototype = self.make_prototype(unlabelled_samples)
        negative_samples = []
        processed_positive_prototype = [v if v > rocchio_threshold else 0 for v in positive_prototype]

        def compare_distance(l):
            values = l.values
            if (np.linalg.norm(values[3:] - positive_prototype) <= 1.1 * np.linalg.norm(
                    values[3:] - unlabelled_prototype)) and (
                    np.linalg.norm(values[3:] - processed_positive_prototype) >
                    np.linalg.norm(values[3:] - unlabelled_prototype)):
                negative_samples.append(values[0])

        if negatives:
            self.onehot[self.onehot['song_id'].isin(unlabelled_samples)].apply(compare_distance, axis=1)
        return negative_samples, {}

    def get_new_samples_1dnf(self, negatives=True, positives=False, strong_positive_threshold=0.3):
        assert (not positives)
        negative_samples = []

        unlabelled_samples = list(set(self.onehot.song_id) - set(self.positive_samples))
        strong_positive_features_indices = [i for i in range(len(self.prototype)) if
                                            self.prototype[i] >= strong_positive_threshold]

        def filter_data(l):
            values = l.values
            if all([values[3:][i] == 0 for i in strong_positive_features_indices]):
                negative_samples.append(values[0])

        if negatives:
            self.onehot[self.onehot['song_id'].isin(unlabelled_samples)].apply(filter_data, axis=1)
        return negative_samples, {}

    def get_likelihood_all_songs(self):
        likelihood_all_songs = pd.DataFrame()
        indices = [0] + [len(self.encoding[k]) for k in self.encoding]
        indices = [sum(indices[:i]) for i in range(1, len(indices) + 1)]
        likelihoods = np.array([1] * self.onehot.shape[0])
        onehot_prototype = self.onehot.iloc[:, 3:].mul(self.prototype)

        for k in range(1, len(indices)):
            likelihoods = likelihoods * np.array(
                onehot_prototype.iloc[:, indices[k - 1]:indices[k]].sum(axis=1))

        likelihood_all_songs['song_id'] = self.onehot.iloc[:, 0]
        likelihood_all_songs['likelihood'] = likelihoods
        likelihood_all_songs = likelihood_all_songs.sort_values(by=['likelihood'], ascending=False)
        likelihood_all_songs.reset_index(inplace=True, drop=True)
        del indices, onehot_prototype
        return likelihood_all_songs

    def score(self, all_context_songs):
        return {'most-general': self._score_most_general_query(all_context_songs),
                'songs-query': self._score_songs_query(all_context_songs),
                'dt-query': self._score_dt_query(all_context_songs)}

    def _score_dt_query(self, all_context_songs):
        fetched_songs = QueryLearner.get_songs_from_query(self.learned_queries['dt'], query_type='dt')
        return evaluator(predicted=fetched_songs, target=all_context_songs)

    def _score_songs_query(self, all_context_songs):
        fetched_songs = QueryLearner.get_songs_from_query(self.learned_queries['songs'], query_type='songs')
        return evaluator(predicted=fetched_songs, target=all_context_songs)

    def _score_most_general_query(self, all_context_songs):
        fetched_songs = QueryLearner.get_songs_from_query({'most-general': self.learned_queries['most-general']},
                                                          query_type='songs')
        return evaluator(predicted=fetched_songs, target=all_context_songs)

    @staticmethod
    def get_songs_from_query(query, query_type=None):
        if query_type == 'dt':
            return get_songs_from_query_dt_rules(query)
        elif query_type == 'songs':
            return get_songs_from_query_song_query(query)
        else:
            raise ValueError('invalid query type: {}'.format(query_type))


class ContextLearner:
    def __init__(self):
        self.n_clusters = None
        self.labels = []
        self.silhouette_scores = None
        self.learned_cluster = {}
        self.learned_contexts = {}

    def learn_clusters(self, X, n_clusters=None, weights=None):
        model = KMeans(n_clusters=n_clusters)
        model.fit(X, sample_weight=weights)

        self.labels = model.labels_
        self.silhouette_scores = silhouette_score(X, model.labels_)

    def make_clusters(self, playlists):
        for i in range(len(self.labels)):
            if self.labels[i] in self.learned_cluster.keys():
                self.learned_cluster[self.labels[i]] = self.learned_cluster[self.labels[i]].union(
                    set(playlists[i].onehot.song_id))
            else:
                self.learned_cluster[self.labels[i]] = set(playlists[i].onehot.song_id)

    def make_queries(self, discard_threshold=0.1, reliable_negative_method='l', rocchio_threshold=0):
        for k in self.learned_cluster:
            query_learner = QueryLearner()
            query_learner.fit(list(self.learned_cluster[k]), onehot=songs, discard_threshold=discard_threshold,
                              reliable_negative_method=reliable_negative_method, rocchio_threshold=rocchio_threshold)
            self.learned_contexts[k] = query_learner

    def fit(self, playlists=None, n_clusters=None, discard_threshold=0.1, reliable_negative_method='l',
            rocchio_threshold=0):
        X = np.array([p.prototype for p in playlists])
        weights = [c.onehot.shape[0] for c in playlists]
        weights = np.array([w / sum(weights) for w in weights])

        self.learn_clusters(X, n_clusters, weights)
        self.make_clusters(playlists)
        self.make_queries(discard_threshold=discard_threshold, reliable_negative_method=reliable_negative_method,
                          rocchio_threshold=rocchio_threshold)

