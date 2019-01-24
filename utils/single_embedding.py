import numpy as np
from logging import getLogger
from collections import defaultdict

from gensim.models import KeyedVectors
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from utils import get_len, smartdict_init, list_of_strings_to_list_of_lists, _pprint

logger = getLogger(__name__)


class _SingleEmbeddingBaseVectorizer(object):
    """
    A base class for single embedding vectorizers.

    A class inheriting from this base class should implement the following methods:
        __repr__
        get_params
        fit
        transform
        fit_transform
    """
    def __init__(self, word2vec, clusters={}, vocabulary=None, norm=None):
        self._is_vocabulary = False
        self._is_clusters = False
        self._is_gensim = False
        self._is_norm = False

        if isinstance(word2vec, dict):
            val = next(iter(word2vec.values()))
            self.dim = get_len(val)
            self.num_vectors = len(word2vec)
            self.emb_vocab = set(word2vec.keys())

        # elif isinstance(word2vec, KeyedVectors) or isinstance(word2vec, FastText):
        else:
            self.dim = word2vec.vector_size
            self.num_vectors = len(word2vec.vocab)
            self.emb_vocab = set(word2vec.vocab.keys())
            self._is_gensim = True
        # else:
        #    raise TypeError("unsupported word2vec type")

        self.word2vec = word2vec

        if isinstance(clusters, dict) and all([isinstance(v, list) for v in clusters.values()]):
            self.clusters = clusters
            if len(clusters):
                self._is_clusters = True
        else:
            raise TypeError("clusters must be of type dictionary, with all values as lists")

        if vocabulary is None:
            self.vocabulary = None
        elif isinstance(vocabulary, (list, set)):
            cluster_tokens = set([k for k in clusters.keys()])

            # the final vocabulary is the intersection of the embedding vocab
            # and the union of pre-loaded vocabulary and cluster-tokens
            # (`vocabulary` OR `cluster_tokens`) AND `emb_vocab`
            self.vocabulary = set(vocabulary).union(cluster_tokens).intersection(self.emb_vocab)
            self._is_vocabulary = True
        else:
            raise ValueError("vocabulary must be either None or of type {list, set}")

        self.mapper = None
        self.clusters_vectors = None
        if self._is_clusters:
            self._build_mapper()
            self._build_vectors_for_clusters()

        if norm is None or not norm:
            self.norm = None
        elif norm in ('l1', 'l2', 'max'):
            self.norm = norm
            self._is_norm = True
        else:
            raise ValueError("unsupported norm value '{}'".format(norm))

        self.base_params = {
            "dim": self.dim,
            "num_vectors": self.num_vectors,
            "clusters": self.clusters,
            "vocabulary": self.vocabulary,
            "norm": self.norm
        }

    def _build_mapper(self):
        """ creates a smart dictionary that will be used for mapping words to cluster-tokens """
        d = dict()
        for cluster_token, word_list in self.clusters.items():
            for word in word_list:
                d[word] = cluster_token

        self.mapper = smartdict_init(d, lambda x: x)

    def _build_vectors_for_clusters(self):
        """ Builds vectors for cluster tokens """

        def _build_single_average_vector(words, w2v, vocab):
            if vocab:
                v = np.mean([w2v[w] for w in words if w in w2v and w in vocab]
                            or [np.zeros(self.dim)], axis=0)
            else:
                v = np.mean([w2v[w] for w in words if w in w2v]
                            or [np.zeros(self.dim)], axis=0)

            return v

        for cluster_token, word_list in self.clusters.items():

            # calculate cluster vector as the average of its words
            vec = _build_single_average_vector(word_list, self.word2vec, self.vocabulary)

            # add cluster vector to embeddings
            if self._is_gensim:
                self.word2vec.add([cluster_token], [vec], replace=True)
            else:
                self.word2vec[cluster_token] = vec

    def _map(self, X):
        """ replaces words with word-clusters
        each word that is in a cluster should be replaced with the cluster token
        each word that is not in a cluster should remain as is
        """
        return [" ".join([self.mapper[e] for e in s.split()]) for s in X]


class MeanEmbeddingVectorizer(_SingleEmbeddingBaseVectorizer):

    def __init__(self, word2vec, clusters={}, vocabulary=None, norm=None):

        # base constructor
        super(MeanEmbeddingVectorizer, self).__init__(word2vec, clusters, vocabulary, norm)

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(), offset=len(class_name), ),)

    def get_params(self):
        out = self.base_params

        return out

    def get_feature_names(self):
        return list(range(self.dim))

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if not get_len(X):
            # X is an empty of elements, return an empty np.ndarray
            return np.empty((0, self.dim))

        X_mapped = self._map(X) if self._is_clusters else X

        seqs = list_of_strings_to_list_of_lists(X_mapped) if isinstance(X_mapped[0], str) else X_mapped

        res = None

        if self._is_vocabulary:
            res = np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec and w in self.vocabulary]
                        or [np.zeros(self.dim)], axis=0)
                for words in seqs
            ])
        else:
            res = np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in seqs
            ])

        if self._is_norm:
            res = normalize(res, norm=self.norm, axis=1)

        return res

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)


class TfidfEmbeddingVectorizer(_SingleEmbeddingBaseVectorizer):

    def __init__(self, word2vec, clusters={}, vocabulary=None, norm=None, missing_val='max_idf', max_features=None,
                 ngram_range=(1, 1), stop_words=None, token_pattern=r"(?u)\b\w[\w']+\b", analyzer='word', max_df=1.0,
                 min_df=1):

        # base constructor
        super(TfidfEmbeddingVectorizer, self).__init__(word2vec, clusters, vocabulary, norm)

        # tf-idf related initialization
        assert missing_val in ['max_idf', 'avg_idf', 'zero']
        self.missing_val = missing_val

        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words,
                                     token_pattern=token_pattern, analyzer=analyzer, vocabulary=self.vocabulary,
                                     max_df=max_df, min_df=min_df)

        self.word2weight = None

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(), offset=len(class_name), ),)

    def get_params(self, deep=False):
        """out = {
            **self.base_params,
            **self.tfidf.get_params(deep=deep)
        }"""
        out = {}

        return out

    def get_feature_names(self):
        return list(range(self.dim))

    def fit(self, X, y=None):

        X_mapped = self._map(X) if self._is_clusters else X

        tfidf = self.tfidf
        tfidf.fit(X_mapped)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        # a second option is taking the average idf  value
        # a third strategy is to give unseen words zero weight
        # and practically ignore them

        default = max(tfidf.idf_) if self.missing_val == 'max_idf' \
            else np.mean(tfidf.idf_) if self.missing_val == 'avg_idf' else 0.0

        self.word2weight = defaultdict(lambda: default, {w: tfidf.idf_[i] for w, i in tfidf.vocabulary_.items()})

        return self

    def transform(self, X):

        if not get_len(X):
            # X is an empty of elements, return an empty np.ndarray
            return np.empty((0, self.dim))

        X_mapped = self._map(X) if self._is_clusters else X

        seqs = list_of_strings_to_list_of_lists(X_mapped) if isinstance(X_mapped[0], str) else X_mapped

        res = None

        if self._is_vocabulary:
            res = np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec and w in self.vocabulary] or
                        [np.zeros(self.dim)], axis=0)
                for words in seqs
            ])
        else:
            res = np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in seqs
            ])

        if self._is_norm:
            res = normalize(res, norm=self.norm, axis=1)

        return res

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)


class DictWeightedEmbeddingVectorizer(_SingleEmbeddingBaseVectorizer):
    """
    Given a dictionary of words and their weights, represents a document as
    its weighted word vector. Words that are not in the dictionary will
    get a default weight.

    :param
        weights    dictionary of the structure {word: weight}
        default    default value to assign to words that are not in `word2weight`
    """

    def __init__(self, word2vec, clusters={}, vocabulary=None, norm=None, weights={}, default=1):

        # base constructor
        super(DictWeightedEmbeddingVectorizer, self).__init__(word2vec, clusters, vocabulary, norm)

        # dict-weighting related initialization
        if isinstance(weights, dict) and\
                all([isinstance(k, str) and isinstance(v, float) for k, v in weights.items()]):
            self.weights = weights
        else:
            raise TypeError("weights must be of type dictionary, with all keys as strings and values floats")

        if isinstance(default, float):
            self.default = default
        else:
            raise ValueError("muse provide a float type default weight. given: {}".format(default))

        # update base vocabulary with words in the weights dictionary
        if self._is_vocabulary:
            self.vocabulary = self.vocabulary.union(set(weights.keys()))

        # log intersection between weights and embedding
        logger.debug("[vocab] embedding: {}, weights: {}, intersection: {}"
                     .format(len(self.emb_vocab), len(weights), len(self.emb_vocab.intersection(set(weights.keys())))))

        self.word2weight = defaultdict(lambda: default, {word: weight for word, weight in weights.items()})

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(), offset=len(class_name), ),)

    def get_params(self):
        """out = {
            **self.base_params,
            'weights': self.weights,
            'default': self.default
        }"""
        out = {}

        return out

    def get_feature_names(self):
        return list(range(self.dim))

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if not get_len(X):
            # X is an empty of elements, return an empty np.ndarray
            return np.empty((0, self.dim))

        X_mapped = self._map(X) if self._is_clusters else X

        seqs = list_of_strings_to_list_of_lists(X_mapped) if isinstance(X_mapped[0], str) else X_mapped

        res = None

        if self._is_vocabulary:
            res = np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec and w in self.vocabulary] or
                        [np.zeros(self.dim)], axis=0)
                for words in seqs
            ])
        else:
            res = np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in seqs
            ])

        if self._is_norm:
            res = normalize(res, norm=self.norm, axis=1)

        return res

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)
