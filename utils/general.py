import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .utils import smartdict_init, get_len, _pprint


class _GeneralBaseVectorizer(object):
    def __init__(self, clusters, vocabulary, dim):
        """
        A base class for general vectorizers.

        A class inheriting from this base class should implement the following methods:
            __repr__
            get_params
            fit
            transform
            fit_transform
        """

        self._is_vocabulary = False
        self._is_clusters = False
        self.mapper = None

        if isinstance(clusters, dict) and all([isinstance(v, list) for v in clusters.values()]):
            self.clusters = clusters
            if len(clusters):
                self._build_mapper()
                self._is_clusters = True
        else:
            raise TypeError("clusters must be of type dictionary, with all values as lists")

        if vocabulary is None:
            self.vocabulary = None
        elif isinstance(vocabulary, (list, set)):
            cluster_tokens = set([k for k in clusters.keys()])

            # the final vocabulary is the union of the
            # pre-loaded vocabulary and cluster-tokens
            # (`vocabulary` OR `cluster_tokens`)
            self.vocabulary = set(vocabulary).union(cluster_tokens)
            self._is_vocabulary = True
        else:
            raise ValueError("vocabulary must be either None or of type {list, set}")

        self.dim = dim

        self.base_params = {
            "clusters": self.clusters,
            "vocabulary": self.vocabulary,
            "dim": self.dim
        }

    def _build_mapper(self):
        """ creates a smart dictionary that will be used for mapping words to cluster-tokens """
        d = dict()
        for cluster_token, word_list in self.clusters.items():
            for word in word_list:
                d[word] = cluster_token

        self.mapper = smartdict_init(d, lambda x: x)

    def _map(self, X):
        """ replaces words with word-clusters
        each word that is in a cluster should be replaced with the cluster token
        each word that is not in a cluster should remain as is
        """
        return [" ".join([self.mapper[e] for e in s.split()]) for s in X]


class GeneralCountVectorizer(_GeneralBaseVectorizer):

    def __init__(self, clusters={}, vocabulary=None,
                 max_features=None, ngram_range=(1, 1), stop_words=None,
                 token_pattern=r"(?u)\b\w[\w']+\b",
                 analyzer='word', max_df=1.0, min_df=1, dim=100):

        # base constructor
        super(GeneralCountVectorizer, self).__init__(clusters, vocabulary, dim)

        # count vectorizer
        self.vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words,
                                   token_pattern=token_pattern, analyzer=analyzer, vocabulary=self.vocabulary,
                                   max_df=max_df, min_df=min_df)

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False), offset=len(class_name),),)

    def get_params(self, deep=False):
        out= {}
        """out = {**self.base_params,
               **self.vec.get_params(deep=deep)}"""

        return out

    def get_feature_names(self):
        return self.vec.get_feature_names()

    def get_stop_words(self):
        return self.vec.get_stop_words()

    def fit(self, X, y=None):

        X_mapped = self._map(X) if self._is_clusters else X

        self.vec.fit(X_mapped)

        return self

    def transform(self, X):

        if not get_len(X):
            # X is an empty of elements, return an empty np.ndarray
            return np.empty((0, self.dim))

        X_mapped = self._map(X) if self._is_clusters else X
        return self.vec.transform(X_mapped)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, X):
        return self.vec.inverse_transform(X)


class GeneralTfidfVectorizer(_GeneralBaseVectorizer):

    def __init__(self, clusters={}, vocabulary=None,
                 max_features=None, ngram_range=(1, 1), stop_words=None,
                 token_pattern=r"(?u)\b\w[\w']+\b",
                 analyzer='word', max_df=1.0, min_df=1, dim=100):

        # base constructor
        super(GeneralTfidfVectorizer, self).__init__(clusters, vocabulary, dim)

        # tfidf vectorizer
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words,
                                   token_pattern=token_pattern, analyzer=analyzer, vocabulary=self.vocabulary,
                                   max_df=max_df, min_df=min_df)

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False), offset=len(class_name), ),)

    def get_params(self, deep=False):
        out = {}
        # out = {**self.base_params,
        #        **self.vec.get_params(deep=deep)}

        return out

    def get_feature_names(self):
        return self.vec.get_feature_names()

    def get_stop_words(self):
        return self.vec.get_stop_words()

    def fit(self, X, y=None):

        X_mapped = self._map(X) if self._is_clusters else X

        self.vec.fit(X_mapped)

        return self

    def transform(self, X):

        if not get_len(X):
            # X is an empty of elements, return an empty np.ndarray
            return np.empty((0, self.dim))

        X_mapped = self._map(X) if self._is_clusters else X
        return self.vec.transform(X_mapped)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, X):
        return self.vec.inverse_transform(X)
