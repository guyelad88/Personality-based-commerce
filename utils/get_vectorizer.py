from general import *
from single_embedding import *


def get_vectorizer(dict_vec, embds_dict, weights={}, weight_default=1, word_clusters={}, vocab_preloaded=None):
    if dict_vec['vec_type'] == 'vec_count':
        vec = GeneralCountVectorizer(
            clusters=word_clusters,
            vocabulary=vocab_preloaded,
            max_features=dict_vec['max_features'],
            ngram_range=dict_vec['ngram_range'],
            stop_words=dict_vec['stop_words'],
            token_pattern=r"(?u)\b\w[\w']+\b",
            analyzer='word',
            max_df=dict_vec['max_df'],
            min_df=dict_vec['min_df'])

    elif dict_vec['vec_type'] == 'vec_tfidf':
        vec = GeneralTfidfVectorizer(
            clusters=word_clusters,
            vocabulary=vocab_preloaded,
            max_features=dict_vec['max_features'],
            ngram_range=dict_vec['ngram_range'],
            stop_words=dict_vec['stop_words'],
            token_pattern=r"(?u)\b\w[\w']+\b",
            analyzer='word',
            max_df=dict_vec['max_df'],
            min_df=dict_vec['min_df'])

    elif dict_vec['vec_type'] == 'vec_avg_embds':
        vec = MeanEmbeddingVectorizer(
            word2vec=embds_dict,
            clusters=word_clusters,
            vocabulary=vocab_preloaded,
            norm=dict_vec['norm'])

    elif dict_vec['vec_type'] == 'vec_tfidf_embeds':
        vec = TfidfEmbeddingVectorizer(
            word2vec=embds_dict,
            clusters=word_clusters,
            vocabulary=vocab_preloaded,
            norm=dict_vec['norm'],
            missing_val=dict_vec['missing_val'],
            max_features=dict_vec['max_features'],
            ngram_range=dict_vec['ngram_range'],
            stop_words=dict_vec['stop_words'],
            token_pattern=r"(?u)\b\w[\w']+\b",
            analyzer='word',
            max_df=dict_vec['max_df'],
            min_df=dict_vec['min_df'])

    else:
        raise ValueError("unsupported vectorizer type")

    return vec
