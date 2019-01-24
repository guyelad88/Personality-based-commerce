from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os

from utils.logger import Logger
from config import bfi_config
from build_item_aspect_feature import BuildItemAspectScore

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn import linear_model, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from utils.get_vectorizer import *
from utils.single_embedding import MeanEmbeddingVectorizer
from sklearn.feature_extraction import stop_words

from eli5 import formatters, explain_weights_xgboost, explain_prediction_xgboost, show_weights
from eli5.sklearn import explain_linear_classifier_weights
from eli5 import sklearn
from scipy.sparse import csr_matrix


class CalculateScore:

    def __init__(self, participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
                 threshold_purchase, bool_slice_gap_percentile=True, bool_normalize_features=True, C=2,
                 cur_penalty='l1', time_purchase_ratio_feature_flag=True, time_purchase_meta_feature_flag=True,
                 vertical_ratio_feature_flag=True, meta_category_feature_flag = True,
                 purchase_percentile_feature_flag=True,
                 user_meta_feature_flag=True, aspect_feature_flag=True, h_limit=0.6, l_limit=0.4,
                 k_best=10, plot_directory='', user_type='all', normalize_traits=True, classifier_type='xgb',
                 split_bool=False, xgb_c=1, xgb_eta=0.1, xgb_max_depth=4, dir_logistic_results='', cur_time='',
                 k_best_feature_flag=True, kv=None):

        self.kv = kv

        # file arguments
        self.participant_file = participant_file
        self.item_aspects_file = item_aspects_file
        self.purchase_history_file = purchase_history_file
        self.valid_users_file = valid_users_file
        self.dir_analyze_name = dir_analyze_name
        self.dir_logistic_results = dir_logistic_results
        self.cur_time = cur_time

        # define data frame needed for analyzing data
        self.participant_df = pd.DataFrame()
        self.item_aspects_df = pd.DataFrame()
        self.purchase_history_df = pd.DataFrame()
        self.valid_users_df = pd.DataFrame()

        self.merge_df = pd.DataFrame()                  # merge df - final for analyze and correlation
        self.raw_df = pd.DataFrame()                    # raw data df using in prediction function

        self.avg_openness, self.avg_conscientiousness, self.avg_extraversion, self.avg_agreeableness, \
            self.avg_neuroticism = 0, 0, 0, 0, 0

        self.ratio_hundred_openness, self.ratio_hundred_conscientiousness, self.ratio_hundred_extraversion, \
            self.ratio_hundred_agreeableness, self.ratio_hundred_neuroticism = 0, 0, 0, 0, 0

        self.question_openness = bfi_config.bfi_test_information['question_openness']
        self.question_conscientiousness = bfi_config.bfi_test_information['question_openness']
        self.question_extraversion = bfi_config.bfi_test_information['question_openness']
        self.question_agreeableness = bfi_config.bfi_test_information['question_openness']
        self.question_neuroticism = bfi_config.bfi_test_information['question_neuroticism']

        # help to calculate percentile
        self.openness_score_list = list()
        self.conscientiousness_score_list = list()
        self.extraversion_score_list = list()
        self.agreeableness_score_list = list()
        self.neuroticism_score_list = list()

        self.min_cost_list = list()
        self.q1_cost_list = list()
        self.median_cost_list = list()
        self.q3_cost_list = list()
        self.max_cost_list = list()
        self.valid_user_list = list()       # valid users
        self.lr_x_feature = list()

        self.item_buyer_dict = dict()       # item-buyer dict 1:N
        self.user_name_id_dict = dict()     # missing data because it 1:N
        self.user_id_name_dict = dict()     #

        self.models_results = list()            # store model result (later will insert into a result CSV)

        self.titles_features_name = list()          # contain title features names
        self.description_features_name = list()     # contain description features names

        # system hyper_parameter
        self.threshold_purchase = threshold_purchase    # throw below this number
        self.C = C
        self.penalty = cur_penalty
        self.bool_slice_gap_percentile = bool_slice_gap_percentile
        self.bool_normalize_features = bool_normalize_features
        self.threshold_pearson = 0.2
        self.test_fraction = 0.2
        self.h_limit = h_limit
        self.l_limit = l_limit

        self.k_best_feature_flag = k_best_feature_flag
        #self.k_rand_min, self.k_rand_max = \
        #    bfi_config.predict_trait_configs['k_rand'][0], bfi_config.predict_trait_configs['k_rand'][1]
        # self.k_best = random.randint(self.k_rand_min, self.k_rand_max)     # number of k_best feature to select / 'all'
        self.k_best = k_best
        self.plot_directory = plot_directory
        self.user_type = user_type                  # user type in model 'all'/'cf'/'ebay-tech'
        self.normalize_traits = normalize_traits    # normalize each trait to 0-1
        self.classifier_type = classifier_type
        self.split_bool = split_bool                # run CV or just test-train
        self.xgb_c = xgb_c
        self.xgb_eta = xgb_eta
        self.xgb_max_depth = xgb_max_depth
        self.xgb_n_estimators = random.randint(300, 1000)
        self.xgb_subsample = round(random.uniform(0.6, 1), 2)
        self.xgb_colsample_bytree = round(random.uniform(0.6, 1), 2)

        self.n_splits = bfi_config.predict_trait_configs['num_splits']
        self.pearson_relevant_feature = bfi_config.feature_data_set['pearson_relevant_feature']
        self.categ_threshold = bfi_config.feature_data_set['categ_threshold']
        self.lr_y_feature = bfi_config.feature_data_set['lr_y_feature']
        self.lr_y_logistic_feature = bfi_config.feature_data_set['lr_y_logistic_feature']
        self.lr_y_linear_feature = bfi_config.feature_data_set['lr_y_linear_feature']
        self.trait_percentile = bfi_config.feature_data_set['trait_percentile']
        self.map_dict_percentile_group = bfi_config.feature_data_set['map_dict_percentile_group']

        self.time_purchase_ratio_feature_flag = time_purchase_ratio_feature_flag
        self.time_purchase_meta_feature_flag = time_purchase_meta_feature_flag
        self.vertical_ratio_feature_flag = vertical_ratio_feature_flag
        self.meta_category_feature_flag = meta_category_feature_flag
        self.purchase_price_feature_flag = False                        # if true is overlap with purchase percentile
        self.purchase_percentile_feature_flag = purchase_percentile_feature_flag
        self.user_meta_feature_flag = user_meta_feature_flag
        self.aspect_feature_flag = aspect_feature_flag
        self.title_feature_flag = bfi_config.predict_trait_configs['dict_feature_flag']['title_feature_flag']
        self.descriptions_feature_flag = bfi_config.predict_trait_configs['dict_feature_flag']['descriptions_feature_flag']

        self.time_purchase_ratio_feature = bfi_config.feature_data_set['time_purchase_ratio_feature']
        self.time_purchase_meta_feature = bfi_config.feature_data_set['time_purchase_meta_feature']
        self.vertical_ratio_feature = bfi_config.feature_data_set['vertical_ratio_feature']
        self.meta_category_feature = bfi_config.feature_data_set['meta_category_feature']
        self.purchase_price_feature = bfi_config.feature_data_set['purchase_price_feature']
        self.purchase_percentile_feature = bfi_config.feature_data_set['purchase_percentile_feature']
        self.user_meta_feature = bfi_config.feature_data_set['user_meta_feature']
        self.aspect_feature = bfi_config.feature_data_set['aspect_feature']

        self.min_df = bfi_config.predict_trait_configs['min_df']
        self.max_textual_features = bfi_config.predict_trait_configs['max_textual_features']

        self.embedding_dim = bfi_config.predict_trait_configs['embedding_dim']
        self.embedding_limit = bfi_config.predict_trait_configs['embedding_limit']
        self.embedding_type = bfi_config.predict_trait_configs['embedding_type']
        self.dict_vec = bfi_config.predict_trait_configs['dict_vec']

        self.textual_title_file_path = bfi_config.predict_trait_configs['title_corpus']
        self.textual_description_file_path = bfi_config.predict_trait_configs['description_corpus']

        self.black_list = bfi_config.black_list

    # load csv into df
    def load_clean_csv_results(self):

        self.participant_df = pd.read_csv(self.participant_file)
        self.item_aspects_df = pd.read_csv(self.item_aspects_file)
        self.purchase_history_df = pd.read_csv(self.purchase_history_file, error_bad_lines=False)
        self.valid_users_df = pd.read_csv(self.valid_users_file)

    def clean_df(self):
        # use only valid user id
        if 'USER_SLCTD_ID' in list(self.valid_users_df):
            tmp_valid_user_list = list(self.valid_users_df['USER_SLCTD_ID'])
        elif 'eBay site user name' in list(self.valid_users_df):
            tmp_valid_user_list = list(self.valid_users_df['eBay site user name'])
        else:
            raise('unknown field')
        self.valid_user_list = [x for x in tmp_valid_user_list if str(x) != 'nan']

        # extract only valid user name
        for (idx, row_participant) in self.participant_df.iterrows():
            # func = lambda s: s[:1].lower() + s[1:] if s else ''
            lower_first_name = row_participant['eBay site user name'].lower()
            self.participant_df.at[idx, 'eBay site user name'] = lower_first_name

        self.participant_df = self.participant_df[self.participant_df['eBay site user name'].isin(self.valid_user_list)]

        before_cnt = self.participant_df.shape[0]
        self.participant_df = self.participant_df[~self.participant_df['eBay site user name'].isin(self.black_list)]

        Logger.info('Removed from black list (eBay user name): {}-{}={}'.format(
            before_cnt,
            self.participant_df.shape[0],
            before_cnt-self.participant_df.shape[0]))

    def insert_gender_feature(self):

        self.merge_df = self.participant_df.copy()

        self.merge_df['gender'] = \
            np.where(self.merge_df['Gender'] == 'Male', 1, 0)

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_gender.csv')
        Logger.info('')
        Logger.info('add gender feature')
        Logger.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df_gender.csv')
        return

    # remain users regards to user_type variable (eBay/tech/CF)
    def remove_except_cf(self):

        Logger.info('')
        Logger.info('extract user regards to user_type variable ' + str(self.user_type))

        assert self.user_type == 'all'

        if self.user_type not in ['all', 'cf', 'ebay-tech']:
            raise('undefined user_type: ' + str(self.user_type))

        if self.user_type == 'cf':
            Logger.info('Remain only users from CF')
            self.merge_df = self.merge_df.loc[self.merge_df['Site'] == 'CF']
            Logger.info('CF users: ' + str(self.merge_df.shape[0]))
        elif self.user_type == 'ebay-tech':
            Logger.info('Remain only users from eBay and Tech')
            self.merge_df = self.merge_df.loc[self.merge_df['Site'] != 'CF']
            Logger.info('Is users: ' + str(self.merge_df.shape[0]))
        elif self.user_type == 'all':
            Logger.info('Remain all users')
            Logger.info('Is users: ' + str(self.merge_df.shape[0]))
        return

    # create_feature_list
    def create_feature_list(self):

        if len(self.lr_x_feature) > 0:
            return

        if self.time_purchase_meta_feature_flag:
            self.lr_x_feature.extend(self.time_purchase_meta_feature)
        if self.time_purchase_ratio_feature_flag:
            self.lr_x_feature.extend(self.time_purchase_ratio_feature)
        if self.vertical_ratio_feature_flag:
            self.lr_x_feature.extend(self.vertical_ratio_feature)
        if self.meta_category_feature_flag:
            self.lr_x_feature.extend(self.meta_category_feature)
        if self.purchase_price_feature_flag:
            self.lr_x_feature.extend(self.purchase_price_feature)
        if self.purchase_percentile_feature_flag:
            self.lr_x_feature.extend(self.purchase_percentile_feature)
        if self.user_meta_feature_flag:
            self.lr_x_feature.extend(self.user_meta_feature)
        if self.aspect_feature_flag:
            self.lr_x_feature.extend(self.aspect_feature)
        if self.title_feature_flag:
            self.lr_x_feature.extend(self.titles_features_name)
        if self.descriptions_feature_flag:
            self.lr_x_feature.extend(self.description_features_name)

    # extract user history purchase - amount
    # a. connect to number of purchases
    # b. connect to purchase after vertical
    def extract_user_purchase_connection(self):

        user_id_name_dict, histogram_purchase_list = self.insert_purchase_amount_data()
        self._slice_participant_using_threshold(histogram_purchase_list)
        self._insert_purchase_vertical_data(user_id_name_dict)

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df.csv')
        Logger.info('')
        Logger.info('add user purchase connection')
        Logger.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df.csv')

    def insert_meta_category(self):
        # Add eBay meta categories features
        _num_row_before = self.merge_df.shape[0]

        categ_df = pd.read_csv(
            '/Users/gelad/Personality-based-commerce/data/participant_data/purchase_data/thresh={}_user_purchases_per_category.csv'.format(self.categ_threshold)
        )
        # m = pd.merge(self.merge_df, categ_df, left_on='eBay site user name', right_on='buyer_name')
        self.merge_df = pd.merge(
            self.merge_df,
            categ_df,
            left_on=['eBay site user name', 'number_purchase'],
            right_on=['buyer_name', 'buyer_cnt']
        )
        assert _num_row_before == self.merge_df.shape[0]
        return

    def no_number_preprocessor(self, tokens):
        import re
        r = re.sub('(\d)+', 'NUM', tokens.lower())
        # This alternative just removes numbers:
        # r = re.sub('(\d)+', '', tokens.lower())
        return r

    def insert_textual_features_embedding(self, text_type):
        """
        create Mean Embedding vector to each user (regared to his titles/descriptions)
        """
        if text_type not in ['titles', 'descriptions']:
            raise ValueError('text type is not defined: {}'.format(text_type))

        # self.textual_title_file_path = bfi_config.predict_trait_configs['title_corpus']
        # self.textual_description_file_path = bfi_config.predict_trait_configs['description_corpus']
        if text_type == 'titles':
            if not self.title_feature_flag:
                Logger.info('title features not in use - skip')
                return
            else:
                Logger.info('Start to extract title features...')

            _df_t = pd.read_csv(
                self.textual_title_file_path,
                usecols=['buyer_id', text_type, 'cnt']
            )
        elif text_type == 'descriptions':
            if not self.descriptions_feature_flag:
                Logger.info('descriptions features not in use - skip')
                return
            else:
                Logger.info('Start to extract descriptions features...')

            _df_t = pd.read_csv(
                self.textual_description_file_path,
                usecols=['buyer_id', text_type, 'cnt']
            )

        _df_t = _df_t.loc[_df_t['buyer_id'].isin(list(self.merge_df['buyer_id']))]
        if text_type == 'titles':
            assert _df_t.shape[0] == self.merge_df.shape[0]

        cv = MeanEmbeddingVectorizer(word2vec=self.kv, norm='l2')
        Logger.info('EMBEDDING DETAILS: type: {}, dim: {}, limit: {}'.format(
            self.embedding_dim,
            self.embedding_limit,
            self.embedding_type
        ))

        texts = _df_t[text_type]        # load relevant column: text_type=titles/descriptions
        texts = texts.str.lower()
        stop_words_en = stop_words.ENGLISH_STOP_WORDS
        texts = texts.apply(lambda x: [item for item in x.split() if item not in stop_words_en])  # TODO cut by min IDF

        texts = texts.tolist()
        X_titles_un_normalized = cv.fit_transform(texts)
        X_titles = X_titles_un_normalized
        print(X_titles.shape[1])

        aa = pd.DataFrame(
            columns=['buyer_id', 'cv_{}'.format(text_type)]
        )

        for i, row in self.merge_df.iterrows():
            if text_type == 'titles':
                row_idx = np.where(_df_t['buyer_id'] == row['buyer_id'])[0][0]
                row_ndarray = X_titles[row_idx]
            elif text_type == 'descriptions':
                temp_row = np.where(_df_t['buyer_id'] == row['buyer_id'])[0]
                if len(temp_row) == 0:
                    continue
                    row_ndarray = np.zeros(X_titles.shape[1], dtype=int)
                elif len(temp_row) == 1:
                    row_ndarray = X_titles[temp_row[0]]
                else:
                    raise('')
            else:
                raise('')

            aa = aa.append({
                'buyer_id': row['buyer_id'],
                'cv_{}'.format(text_type): row_ndarray
            }, ignore_index=True)

        if text_type == 'titles':
            assert aa.shape[0] == self.merge_df.shape[0]

        bb = pd.merge(
            self.merge_df,
            aa,
            left_on=['buyer_id'],
            right_on=['buyer_id'])

        if text_type == 'titles':
            assert bb.shape[0] == self.merge_df.shape[0]

        f_name_raw = cv.get_feature_names()

        if text_type == 'titles':
            self.titles_features_name = ['{}_{}'.format(text_type, f_n) for f_n in f_name_raw]
            df3 = pd.DataFrame(bb['cv_{}'.format(text_type)].values.tolist(), columns=self.titles_features_name)

        elif text_type == 'descriptions':
            self.description_features_name = ['{}_{}'.format(text_type, f_n) for f_n in f_name_raw]
            df3 = pd.DataFrame(bb['cv_{}'.format(text_type)].values.tolist(), columns=self.description_features_name)

        else:
            raise ValueError('unknown text type: {}'.format(text_type))

        if text_type == 'titles':
            before_shape = self.merge_df.shape[0]
            self.merge_df = pd.concat([self.merge_df, df3], axis=1)
            assert self.merge_df.shape[0] == before_shape
        elif text_type == 'descriptions':
            bb = bb.drop(['cv_{}'.format(text_type)], axis=1)
            before_shape = bb.shape[0]
            self.merge_df = pd.concat([bb, df3], axis=1)
            assert self.merge_df.shape[0] == before_shape

        Logger.info('finish to insert textual features')
        Logger.info('CountVectorizer properties: {}'.format(cv))

    def _append_textual_features(self, _X_train, _X_test):
        """
        steps:
            1. load objects
                a. textual data
                b. kv
                c. params
            1. define vectorizer
            2. fit_transform on train
            3. transform on test
            4. return
        :param _X_train:
        :param _X_test:
        :return:
        """
        if not self.title_feature_flag:
            Logger.info('titles textual feature not in use')
            return _X_train, _X_test, list(_X_train)

        Logger.info('Before added titles features: train: {} test: {}'.format(_X_train.shape, _X_test.shape))

        _df_purchase_titles = pd.read_csv(
            '/Users/gelad/Personality-based-commerce/data/participant_data/purchase_data/titles_corpus_710.csv',
            usecols=['buyer_id', 'titles', 'cnt']
        )
        all_buyers_id = list(_X_train['buyer_id']) + list(_X_test['buyer_id'])

        _df_train = pd.merge(_df_purchase_titles, _X_train, on='buyer_id')
        _df_test = pd.merge(_df_purchase_titles, _X_test, on='buyer_id')

        assert _X_train.shape[0] == _df_train.shape[0]
        assert _X_test.shape[0] == _df_test.shape[0]

        # vec = MeanEmbeddingVectorizer(word2vec=self.kv, norm='l2')
        vec = get_vectorizer(self.dict_vec, self.kv)
        Logger.info('VECTORIZER TYPE: {}, max features: {}'.format(self.dict_vec['vec_type'], self.dict_vec['max_features']))
        Logger.info('EMBEDDING DETAILS: type: {}, dim: {}, limit: {}'.format(
            self.embedding_type,
            self.embedding_dim,
            self.embedding_limit
        ))

        def _prepare_data(_df):
            texts = _df['titles']
            texts = texts.str.lower()
            stop_words_en = stop_words.ENGLISH_STOP_WORDS
            texts = texts.apply(lambda x: [item for item in x.split() if item not in stop_words_en])
            texts = texts.apply(lambda x: " ".join(x))
            texts = texts.tolist()
            return texts

        texts_train = _prepare_data(_df_train)
        texts_test = _prepare_data(_df_test)

        assert len(texts_train) == _df_train.shape[0]
        assert len(texts_test) == _df_test.shape[0]

        text_features_train_vectorized = vec.fit_transform(texts_train)
        text_features_test_vectorized = vec.transform(texts_test)

        def _concat(_X, _text_features):
            _text_features = _text_features.todense() if isinstance(_text_features, csr_matrix) else _text_features
            df_vec = pd.DataFrame(_text_features)
            df_vec.columns = ["titles_vec_{}".format(col_name) for col_name in df_vec.columns]

            _X = _X.reset_index()
            _X_final = pd.concat((_X, df_vec), axis=1)

            return _X_final

        _X_train = _concat(_X_train, text_features_train_vectorized)
        _X_test = _concat(_X_test, text_features_test_vectorized)

        # remove unwanted columns
        Logger.info('After added titles features: train: {} test: {}'.format(_X_train.shape, _X_test.shape))
        return _X_train, _X_test, list(_X_train)

    def _pad_description_df(self, _df_desc, _buyer_list):
        """
        auxiliary function - pad users that does not have any descriptions
        """
        input_shape = _df_desc.shape[0]
        for _buyer in _buyer_list:
            if _buyer not in list(_df_desc['buyer_id']):
                _df_desc = _df_desc.append({
                    'buyer_id': _buyer,
                    'cnt': 0,
                    'descriptions': ''
                }, ignore_index=True)

        Logger.info('Input: {}, Output: {}'.format(input_shape, _df_desc.shape[0]))
        return _df_desc

    def insert_descriptions_features(self):
        # add descriptions n-grams features

        if not self.descriptions_feature_flag:
            Logger.info('descriptions features not in use - skip')
            return

        else:
            Logger.info('Start to extract descriptions features...')

        _df_t = pd.read_csv(
            '/Users/gelad/Personality-based-commerce/data/participant_data/purchase_data/descriptions_corpus_438.csv',
            usecols=['buyer_id', 'descriptions', 'cnt']
        )
        # TODO filter by user id/name and remain only relevant id's
        _df_t = _df_t.loc[_df_t['buyer_id'].isin(list(self.merge_df['buyer_id']))]
        _df_t = self._pad_description_df(_df_t, list(self.merge_df['buyer_id']))
        # self.merge_df = self.merge_df.loc[self.merge_df['buyer_id'].isin(list(_df_t['buyer_id']))]
        assert _df_t.shape[0] == self.merge_df.shape[0]
        # normalized CountVectorizer by number of purchases
        # defined diagonal matrix
        cnt_vector = _df_t['cnt']
        # cnt_vector = [1 / float(x) for x in cnt_vector]
        cnt_vector = [1 / float(x) if x > 0 else 0 for x in cnt_vector]
        c = np.array(cnt_vector)
        diag = np.diag(c)

        # defined CountVectorizer properties
        cv_d = CountVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=self.min_df,
            stop_words='english',
            max_features=self.max_textual_features,
            preprocessor=self.no_number_preprocessor
        )
        texts = _df_t['descriptions']
        X_titles_un_normalized = cv_d.fit_transform(texts)

        X_titles = diag * X_titles_un_normalized

        assert X_titles.shape[1] == self.max_textual_features

        aa = pd.DataFrame(
            columns=['buyer_id', 'cv_d']
        )

        for i, row in self.merge_df.iterrows():
            # print(row['buyer_id'])
            row_idx = np.where(_df_t['buyer_id'] == row['buyer_id'])[0][0]
            # print(X_titles[row_idx].shape)
            aa = aa.append({
                'buyer_id': row['buyer_id'],
                'cv_d': X_titles[row_idx]
            }, ignore_index=True)

        assert aa.shape[0] == self.merge_df.shape[0]
        bb = pd.merge(
            self.merge_df,
            aa,
            left_on=['buyer_id'],
            right_on=['buyer_id'])

        assert bb.shape[0] == self.merge_df.shape[0]

        f_name_raw = cv_d.get_feature_names()
        self.description_features_name = ['description_{}'.format(f_n) for f_n in f_name_raw]
        df3 = pd.DataFrame(bb['cv_d'].values.tolist(), columns=self.description_features_name)

        # TODO why assert is fail
        before_shape = self.merge_df.shape[0]
        self.merge_df = pd.concat([self.merge_df, df3], axis=1)
        self.merge_df = self.merge_df[pd.notnull(self.merge_df['Site'])]

        assert self.merge_df.shape[0] == before_shape

        Logger.info('finish to insert description textual features')
        Logger.info('CountVectorizer properties: {}'.format(cv_d))

    # remove participant with purchase amount below threshold
    # visual purchase histogram
    def _slice_participant_using_threshold(self, histogram_purchase_list):
        # remove user buy less than threshold
        before_slice_users = self.merge_df.shape[0]
        self.merge_df = self.merge_df.loc[self.merge_df['number_purchase'] >= self.threshold_purchase]
        Logger.info('')
        Logger.info('Threshold used: ' + str(self.threshold_purchase))
        Logger.info('# participant after slice threshold: ' + str(self.merge_df.shape[0]))
        Logger.info('# participant deleted: ' + str(before_slice_users - self.merge_df.shape[0]))
        Logger.info('# purchases threshold q1: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.25]))
        Logger.info('# purchases threshold median: ' + str(self.merge_df['number_purchase'].median()))
        Logger.info('# purchases threshold q3: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.75]))

        self.merge_df.to_csv(self.dir_analyze_name + 'purchase_amount_after_threshold.csv')
        Logger.info('')
        Logger.info('slice participants below purchase threshold')
        Logger.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'purchase_amount_after_threshold.csv')

        # histogram of number of purchases
        plt.hist(histogram_purchase_list, bins=30)
        plt.title('Histogram of #purchase item per participants, #P ' + str(self.merge_df.shape[0]))
        plt.ylabel('Participant amount')
        plt.xlabel('#Purchases')
        plot_name = self.dir_analyze_name + 'histogram_purchases_per_user' + '_p_' + str(self.merge_df.shape[0]) + '_threshold_' + str(self.threshold_purchase) + '.png'
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()

    # connect to purchase per vertical
    def _insert_purchase_vertical_data(self, user_id_name_dict):

        # plot number of purchase per vertical
        vertical_list = list(self.purchase_history_df['BSNS_VRTCL_NAME'].unique())
        amount_series = self.purchase_history_df['BSNS_VRTCL_NAME'].value_counts()
        Logger.info('Number of purchases: ' + str(len(self.purchase_history_df['BSNS_VRTCL_NAME'])))
        amount_series.plot.bar(figsize=(8, 6))
        plt.title('Vertical vs. Purchase amount')
        plt.ylabel('Purchase Amount')
        plt.xlabel('Vertical')
        plt.xticks(rotation=35)
        plot_name = self.dir_analyze_name + 'vertical_vs_purchases_amount' + '.png'
        plt.savefig(plot_name, bbox_inches='tight')
        # plt.show()
        plt.close()

        # participant amount and ratio per vertical
        for cur_vertical, vertical_amount in amount_series.iteritems():
            Logger.info('Vertical: ' + str(cur_vertical) + ', Amount: ' + str(vertical_amount))
            self.merge_df[str(cur_vertical) + '_amount'] = 0.0
            self.merge_df[str(cur_vertical) + '_ratio'] = 0.0
        # amount and ratio for each vertical
        grouped = self.purchase_history_df.groupby(['buyer_id'])  # groupby how many each user bought
        for name, group in grouped:
            key_type = 'int'
            if key_type == 'int':
                cur_user_name = user_id_name_dict[float(list(group['buyer_id'])[0])]
            else:   # aka key_type == 'str':
                cur_user_name = user_id_name_dict[str(list(group['buyer_id'])[0])]

            # cur_user_name = user_id_name_dict[float(list(group['buyer_id'])[0])]
            # cur_user_name = user_id_name_dict[str(list(group['buyer_id'])[0])]

            # only insert if user in list (74>69 ask Hadas)
            if cur_user_name in list(self.merge_df['eBay site user name']):

                # user row index
                cur_idx = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]

                # a. amount for each vertical
                # b. ratio for each vertical
                # user_id = list(group['buyer_id'])[0]
                # user_name = cur_user_name
                cnt = 0
                group_vertical = group.groupby(['BSNS_VRTCL_NAME'])
                for cur_vertical, vec_df_group in group_vertical:
                    cur_ratio = float(vec_df_group.shape[0])/float(group.shape[0])
                    cnt += vec_df_group.shape[0]
                    self.merge_df.at[cur_idx, str(cur_vertical) + '_amount'] = vec_df_group.shape[0]
                    self.merge_df.at[cur_idx, str(cur_vertical) + '_ratio'] = cur_ratio

        return

    # a. histogram of common aspect, total and per vertical
    # b. insert aspect per item
    def extract_item_aspect(self):
        # a. histogram of common aspect, total and per vertical
        # self.item_aspects_df = pd.read_csv(self.item_aspects_file)
        if not self.aspect_feature_flag:
            Logger.info('item aspects features not in use - skip')
            return
        else:
            Logger.info('Start to extract item aspect feature...')

        item_aspect_obj = BuildItemAspectScore(self.item_aspects_df, self.participant_df, self.purchase_history_df,
                                               self.valid_users_df, self.merge_df, self.user_id_name_dict, self.aspect_feature)
        item_aspect_obj.add_aspect_features()
        self.merge_df = item_aspect_obj.merge_df
        Logger.info('number of features after add item aspect: {}'.format(self.merge_df.shape[1]))

    # normalize trait to 0-1 scale (div by 5)
    def normalize_personality_trait(self):

        Logger.info('')
        Logger.info('normalize flag: ' + str(self.normalize_traits))
        if self.normalize_traits:
            for c_trait in self.lr_y_feature:
                self.merge_df[c_trait] = self.merge_df[c_trait] / 5.0
                Logger.info('normalize trait: {}, Avg: {}, Std: {}'.format(
                    c_trait,
                    round(self.merge_df[c_trait].mean(), 2),
                    round(self.merge_df[c_trait].std(), 2)
                ))

    # calculate traits valuers and percentile per participant
    def cal_participant_traits_values(self):

        # add personality traits empty columns
        self._add_traits_feature_columns()

        # add average traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            Logger.info('Calculate traits value for participant: ' + str(row_participant['Email address']))
            self._calculate_individual_score(idx, row_participant)

        # add percentile traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            # Logger.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self._cal_participant_traits_percentile_values(idx, row_participant)

        # after calculate traits score+percentile extract only relevant features
        '''remain_feature_list = ['Full Name', 'Gender', 'eBay site user name', 'Age', 'openness_trait',
                               'conscientiousness_trait', 'extraversion_trait', 'agreeableness_trait',
                               'neuroticism_trait', 'openness_percentile', 'conscientiousness_percentile',
                               'extraversion_percentile', 'agreeableness_percentile', 'neuroticism_percentile',
                               'age_group']'''

        self.merge_df = self.participant_df.copy()

        # self.merge_df = self.participant_df[remain_feature_list].copy()

        return

    # define personality traits columns in DF
    def _add_traits_feature_columns(self):

        # add empty columns
        self.participant_df["openness_trait"] = np.nan
        self.participant_df["conscientiousness_trait"] = np.nan
        self.participant_df["extraversion_trait"] = np.nan
        self.participant_df["agreeableness_trait"] = np.nan
        self.participant_df["neuroticism_trait"] = np.nan
        self.participant_df["openness_percentile"] = np.nan
        self.participant_df["conscientiousness_percentile"] = np.nan
        self.participant_df["extraversion_percentile"] = np.nan
        self.participant_df["agreeableness_percentile"] = np.nan
        self.participant_df["neuroticism_percentile"] = np.nan
        self.participant_df["age_group"] = ''  # a (15-24), b (25-29), c(30-34), d(35-39), e(40-100)

        return

    # calculate traits values for one participant
    def _calculate_individual_score(self, idx, row_participant):

        op_trait = self.cal_participant_traits(row_participant, self.question_openness,
                                                     self.ratio_hundred_openness)

        self.participant_df.set_value(idx, 'openness_trait', op_trait)
        self.openness_score_list.append(op_trait)

        co_trait = self.cal_participant_traits(row_participant, self.question_conscientiousness,
                                                              self.ratio_hundred_conscientiousness)
        self.participant_df.set_value(idx, 'conscientiousness_trait', co_trait)
        self.conscientiousness_score_list.append(co_trait)

        ex_trait = self.cal_participant_traits(row_participant, self.question_extraversion,
                                                         self.ratio_hundred_extraversion)
        self.participant_df.set_value(idx, 'extraversion_trait', ex_trait)
        self.extraversion_score_list.append(ex_trait)

        ag_trait = self.cal_participant_traits(row_participant, self.question_agreeableness,
                                                          self.ratio_hundred_agreeableness)
        self.participant_df.set_value(idx, 'agreeableness_trait', ag_trait)
        self.agreeableness_score_list.append(ag_trait)

        ne_trait = self.cal_participant_traits(row_participant, self.question_neuroticism,
                                                        self.ratio_hundred_neuroticism)
        self.participant_df.set_value(idx, 'neuroticism_trait', ne_trait)
        self.neuroticism_score_list.append(ne_trait)

        # a (15-24), b (25-29), c(30-34), d(35-39), e(40-100)
        if row_participant['Age'] <= 24:
            self.participant_df.set_value(idx, 'age_group', 'a')
        elif row_participant['Age'] <= 29:
            self.participant_df.set_value(idx, 'age_group', 'b')
        elif row_participant['Age'] <= 34:
            self.participant_df.set_value(idx, 'age_group', 'c')
        elif row_participant['Age'] <= 39:
            self.participant_df.set_value(idx, 'age_group', 'd')
        else:
            self.participant_df.set_value(idx, 'age_group', 'e')
        return

    # calculate percentile value for one participant
    def _cal_participant_traits_percentile_values(self, idx, participant_score):

        op_per = float(sum(i < participant_score['openness_trait'] for i in self.openness_score_list))/float(len(self.openness_score_list)-1)
        self.merge_df.at[idx, 'openness_percentile'] = op_per
        co_per = float(sum(
            i < participant_score['conscientiousness_trait'] for i in self.conscientiousness_score_list))/float(len(self.conscientiousness_score_list)-1)
        self.merge_df.at[idx, 'conscientiousness_percentile'] = co_per
        ex_per = float(sum(
            i < participant_score['extraversion_trait'] for i in self.extraversion_score_list))/float(len(self.extraversion_score_list)-1)
        self.merge_df.at[idx, 'extraversion_percentile'] = ex_per
        ag_per = float(sum(
            i < participant_score['agreeableness_trait'] for i in self.agreeableness_score_list))/float(len(self.agreeableness_score_list)-1)
        self.merge_df.at[idx, 'agreeableness_percentile'] = ag_per
        ne_per = float(sum(
            i < participant_score['neuroticism_trait'] for i in self.neuroticism_score_list))/float(len(self.neuroticism_score_list)-1)
        self.merge_df.at[idx, 'neuroticism_percentile'] = ne_per

        return

    # add price features - value and percentile
    def insert_money_feature(self):

        Logger.info('')
        Logger.info('extract money features')

        self._add_price_feature_columns()
        self._add_price_feature()                # insert value feature
        self._add_percentile_price_feature()     # insert percentile feature

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_cost_value_percentile.csv')
        Logger.info('')
        Logger.info('add cost value percentile features')
        Logger.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df_cost_value_percentile.csv')
        return

    # define money columns in DF
    def _add_price_feature_columns(self):
        self.merge_df['median_purchase_price'] = np.nan
        self.merge_df['q1_purchase_price'] = np.nan
        self.merge_df['q3_purchase_price'] = np.nan
        self.merge_df['min_purchase_price'] = np.nan
        self.merge_df['max_purchase_price'] = np.nan

        self.merge_df['median_purchase_price_percentile'] = np.nan
        self.merge_df['q1_purchase_price_percentile'] = np.nan
        self.merge_df['q3_purchase_price_percentile'] = np.nan
        self.merge_df['min_purchase_price_percentile'] = np.nan
        self.merge_df['max_purchase_price_percentile'] = np.nan
        return

    # find statistical money values per user
    def _add_price_feature(self):
        # Price_USD
        price_group = self.purchase_history_df.groupby(['buyer_id'])
        for buyer_id, group in price_group:
            # print(str(buyer_id) + ': ' + str(group.shape[0]))

            cur_user_name = self.user_id_name_dict[float(buyer_id)]
            # cur_user_name = self.user_id_name_dict[str(buyer_id)]

            if cur_user_name not in self.merge_df['eBay site user name'].tolist():
                continue

            user_percentile_price = group['Price_USD'].quantile([0, .25, .5, 0.75, 1])
            # cur_user_name = self.user_id_name_dict[str(buyer_id)]
            cur_user_name = self.user_id_name_dict[float(buyer_id)]
            cur_idx = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]

            self.merge_df.at[cur_idx, 'min_purchase_price'] = user_percentile_price[0]
            self.min_cost_list.append(user_percentile_price[0])
            self.merge_df.at[cur_idx, 'q1_purchase_price'] = user_percentile_price[0.25]
            self.q1_cost_list.append(user_percentile_price[0.25])
            self.merge_df.at[cur_idx, 'median_purchase_price'] = user_percentile_price[0.5]
            self.median_cost_list.append(user_percentile_price[0.5])
            self.merge_df.at[cur_idx, 'q3_purchase_price'] = user_percentile_price[0.75]
            self.q3_cost_list.append(user_percentile_price[0.75])
            self.merge_df.at[cur_idx, 'max_purchase_price'] = user_percentile_price[1]
            self.max_cost_list.append(user_percentile_price[1])

        return

    # extract percentile values for price features
    def _add_percentile_price_feature(self):

        for (idx, row_participant) in self.merge_df.iterrows():

            min_per = float(sum(i < row_participant['min_purchase_price'] for i in self.min_cost_list)) / float(
                len(self.min_cost_list) - 1)
            # self.merge_df.set_value(idx, 'min_purchase_price_percentile', min_per)
            self.merge_df.at[idx, 'min_purchase_price_percentile'] = min_per

            q1_per = float(sum(i < row_participant['q1_purchase_price'] for i in self.q1_cost_list)) / float(
                len(self.q1_cost_list) - 1)
            self.merge_df.at[idx, 'q1_purchase_price_percentile'] = q1_per

            median_per = float(sum(i < row_participant['median_purchase_price'] for i in self.median_cost_list)) / float(
                len(self.median_cost_list) - 1)
            self.merge_df.at[idx, 'median_purchase_price_percentile'] = median_per

            q3_per = float(sum(i < row_participant['q3_purchase_price'] for i in self.q3_cost_list)) / float(
                len(self.q3_cost_list) - 1)
            self.merge_df.at[idx, 'q3_purchase_price_percentile'] = q3_per

            max_per = float(sum(i < row_participant['max_purchase_price'] for i in self.max_cost_list)) / float(
                len(self.max_cost_list) - 1)
            self.merge_df.at[idx, 'max_purchase_price_percentile'] = max_per
        return

    # add time features
    def insert_time_feature(self):

        self.merge_df['day_ratio'] = 0.0
        self.merge_df['evening_ratio'] = 0.0
        self.merge_df['night_ratio'] = 0.0
        self.merge_df['evening_ratio'] = 0.0
        self.merge_df['first_purchase'] = 0.0
        self.merge_df['last_purchase'] = 0.0
        self.merge_df['tempo_purchase'] = 0.0

        add_per_day = float(1)/float(365)
        import time
        price_group = self.purchase_history_df.groupby(['buyer_id'])
        # self.purchase_history_df['TRX_Timestamp'] = pd.to_datetime(self.purchase_history_df['TRX_Timestamp'])
        # iterate over each user
        for buyer_id, group in price_group:

            # check if user valid + find his nationality + his index in final df
            # cur_user_name = self.user_id_name_dict[str(buyer_id)]
            cur_user_name = self.user_id_name_dict[float(buyer_id)]
            if cur_user_name not in self.merge_df['eBay site user name'].tolist():
                continue
            cur_idx = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]
            cur_nationality = self.merge_df.loc[self.merge_df['eBay site user name'] == cur_user_name].iloc[0][
                'Nationality']
            time_zone_adapt = self._find_time_zone_shift(cur_nationality)        # shift in time zone

            user_count_type = {
                'day': 0,
                'evening': 0,
                'night': 0,
                'weekend': 0,
                'day_ratio': 0,
                'evening_ratio': 0,
                'night_ratio': 0,
                'weekend_ratio': 0,
                'first_year': 3000,
                'first_day': 400,
                'last_day': 0,
                'last_year': 0,
                'tempo': 0,
                'tempo_purchase': 0
            }

            purchase_time_list = group['TRX_Timestamp'].tolist()        # all time user purchase
            cnt_non_weekend = 0
            for cur_per in purchase_time_list:

                # until 8.12
                # time_object = time.strptime(cur_per, '%d/%m/%Y %H:%M')

                # 8.12
                time_object = time.strptime(cur_per[:-2], '%Y-%m-%d %H:%M:%S')
                # cal first/last purchase and tempo
                if time_object.tm_year < user_count_type['first_year'] or (time_object.tm_year == user_count_type['first_year'] and time_object.tm_yday < user_count_type['first_day']):
                    user_count_type['first_year'] = time_object.tm_year
                    user_count_type['first_day'] = time_object.tm_yday

                if time_object.tm_year > user_count_type['last_year'] or (time_object.tm_year == user_count_type['last_year'] and time_object.tm_yday > user_count_type['last_day']):
                    user_count_type['last_year'] = time_object.tm_year
                    user_count_type['last_day'] = time_object.tm_yday

                # insert time in day/week
                # fit time zone
                correct_hour = (time_object.tm_hour + time_zone_adapt) % 24       # DB + shift hour to fit time-zone

                if time_object.tm_hour + time_zone_adapt >= 24:     # if shift change day in week
                    if time_object.tm_wday == 6:
                        tm_wday = 0
                    else:
                        tm_wday = time_object.tm_wday + 1
                else:
                    tm_wday = time_object.tm_wday

                # check weekend/non-weekend
                if cur_nationality == 'Israel':
                    if tm_wday in [4, 5]:
                        user_count_type['weekend'] += 1
                        continue
                else:   # other country
                    if tm_wday in [5, 6]:
                        user_count_type['weekend'] += 1
                        continue

                cnt_non_weekend += 1

                # ration for non-weekend
                if 6 <= correct_hour < 18:
                    user_count_type['day'] += 1
                if 22 >= correct_hour >= 19:
                    user_count_type['evening'] += 1
                if correct_hour >= 23 or correct_hour < 6:
                    user_count_type['night'] += 1

            # print(user_count_type)
            # calculate_ratio

            if cnt_non_weekend > 0:
                user_count_type['day_ratio'] = float(user_count_type['day']) / float(cnt_non_weekend)
                user_count_type['evening_ratio'] = float(user_count_type['evening']) / float(cnt_non_weekend)
                user_count_type['night_ratio'] = float(user_count_type['night']) / float(cnt_non_weekend)
            user_count_type['weekend_ratio'] = float(user_count_type['weekend']) / float(group.shape[0])

            # cal first/last purchase and tempo
            user_count_type['first_purchase'] = float(user_count_type['first_year']) + \
                                                float(add_per_day) * float(user_count_type['first_day'])
            user_count_type['last_purchase'] = float(user_count_type['last_year']) + \
                                                float(add_per_day) * float(user_count_type['last_day'])

            if float(float(user_count_type['last_purchase']) - float(user_count_type['first_purchase'])) != 0:
                user_count_type['tempo_purchase'] = float(group.shape[0])/float(float(user_count_type['last_purchase']) - float(user_count_type['first_purchase']))

            self.merge_df.at[cur_idx, 'day_ratio'] = user_count_type['day_ratio']
            self.merge_df.at[cur_idx, 'evening_ratio'] =  user_count_type['evening_ratio']
            self.merge_df.at[cur_idx, 'night_ratio'] = user_count_type['night_ratio']
            self.merge_df.at[cur_idx, 'weekend_ratio'] = user_count_type['weekend_ratio']
            self.merge_df.at[cur_idx, 'first_purchase'] = user_count_type['first_purchase']
            self.merge_df.at[cur_idx, 'last_purchase'] = user_count_type['last_purchase']
            self.merge_df.at[cur_idx, 'tempo_purchase'] = user_count_type['tempo_purchase']

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_time_purchase.csv')
        Logger.info('')
        Logger.info('add time purchase features')
        Logger.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df_time_purchase.csv')
        return

    # mapping of user country and time zone
    @staticmethod
    def _find_time_zone_shift(country):

        def find_shift(country):        # relative to -7 (server in USA)
            return {
                'Israel': 10,
                'South Africa': 10,
                'Spain': 8,
                'Canada': 3,            # TODO
                'Russia': 11,           # TODO
                'United Kingdom': 8,
                'Italy': 9,
                'United States': 0,     # TODO
                'Germany': 9,
                'Venezuela': 4,
                'Portugal': 7,
                'France': 9,
                'Haiti': 3,
                'Egypt': 10,
                'Turkey': 11,
                'Kazakhstan': 13,
                'Moldova': 10,
                'Philippines': 16,
                'Brazil': 4,
                'Bulgaria': 10,
                'Macau': 16,
                'Serbia': 9,
                'Australia': 17,
                'Angola': 8,
                'Argentina': 3,
                'Albania': 9,
                'China': 16,
                'Paraguay': 5,
                'Slovenia': 9,
                'Peru': 3,
                'Honduras': 2,
                'Netherlands': 9,
                'Algeria': 9,
                'India': 13,
                'Poland': 9,
                'Antigua and Barbuda': 4
            }[country]

        # TODO add countries

        shift = find_shift(country=country)
        return shift

    # connect to number of purchases
    def insert_purchase_amount_data(self):
        self.merge_df['number_purchase'] = np.nan

        self.user_id_name_dict = dict(zip(self.valid_users_df['USER_ID'], self.valid_users_df.USER_SLCTD_ID))
        # for key_id, val_name in self.user_id_name_dict.iteritems():
        for key_id in self.user_id_name_dict.keys():
            if type(key_id) is not int and type(key_id) is not long:
                if not key_id.isdigit():
                    self.user_id_name_dict.pop(key_id, None)    # remove key not valid user id (a number must be)

        self.user_name_id_dict = dict(zip(self.valid_users_df.USER_SLCTD_ID, self.valid_users_df['USER_ID']))
        # for key_name, val_id in self.user_name_id_dict.iteritems():
        for key_id in self.user_name_id_dict.keys():
            if type(self.user_name_id_dict[key_id]) is not int and type(self.user_name_id_dict[key_id]) is not long:
                if not self.user_name_id_dict[key_id].isdigit():
                    self.user_name_id_dict.pop(key_id, None)  # remove key not valid user id (a number must be)
        # from math import isnan
        # self.user_id_name_dict = {k: self.user_id_name_dict[k] for k in self.user_id_name_dict if not isnan(k)}

        # add number of purchase per user
        if type(self.user_id_name_dict.keys()[0]) is int:
            key_type = 'int'
        elif isinstance(type(self.user_id_name_dict.keys()[0]), basestring):
            key_type = 'str'
        else:
            raise ValueError('unknown key type: self.user_id_name_dict.keys()[0]')

        sum = 0
        counter_id = 0
        histogram_purchase_list = list()
        grouped = self.purchase_history_df.groupby(['buyer_id'])  # groupby how many each user bought
        for name, group in grouped:

            if key_type == 'int':
                cur_user_name = self.user_id_name_dict[float(list(group['buyer_id'])[0])]
            else:   # aka key_type == 'str':
                cur_user_name = self.user_id_name_dict[str(list(group['buyer_id'])[0])]

            # only insert if user in list (74 > 69 ask Hadas)
            if cur_user_name in list(self.merge_df['eBay site user name']):
                cur_idx = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]
                self.merge_df.at[cur_idx, 'number_purchase'] = group.shape[0]
                counter_id += 1
                if group.shape[0] > 200:
                    histogram_purchase_list.append(200)
                else:
                    histogram_purchase_list.append(group.shape[0])
                sum += group.shape[0]

        # calculate purchase threshold
        Logger.info('# participant: ' + str(self.merge_df.shape[0]))
        Logger.info('# purchases q1: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.25]))
        Logger.info('# purchases median: ' + str(self.merge_df['number_purchase'].median()))
        Logger.info('# purchases q3: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.75]))

        return self.user_id_name_dict, histogram_purchase_list

    def save_predefined_data_set(self):
        predefined_df_path = os.path.join(self.dir_logistic_results, 'pre_defined_df')
        if not os.path.exists(predefined_df_path):
            os.makedirs(predefined_df_path)

        predefined_df_path = os.path.join(predefined_df_path, 'shape={}_{}_time=_{}.csv'.format(
            self.merge_df.shape[0],
            self.merge_df.shape[1],
            self.cur_time
        ))
        # TODO add content (about slicing: e.g. min purchase amount...)
        self.merge_df.to_csv(predefined_df_path, index=False)
        Logger.info('save pre-defined data set shape: {}'.format(self.merge_df.shape))

    def load_predefined_data_set(self, predefined_path):
        self.merge_df = pd.read_csv(predefined_path)
        Logger.info('load pre-defined data set shape: {}'.format(self.merge_df.shape))

    # calculate logistic regression model
    def calculate_logistic_regression(self):

        # self.map_dict_percentile_group = dict(zip(self.lr_y_logistic_feature,  self.trait_percentile))

        self.models_results = list()     # contain all results for the 5 traits models

        Logger.info('add textual features: {}'.format(len(self.lr_x_feature)))

        relevant_X_columns = copy.deepcopy(self.lr_x_feature)
        # map_dict_feature_non_zero = dict()
        # for trait in self.lr_y_logistic_feature:
        #    map_dict_feature_non_zero[trait] = dict(zip(list(relevant_X_columns), [0]*len(relevant_X_columns)))

        # add column H/L for each trait
        self.add_high_low_traits_column()

        for idx, y_feature in enumerate(self.lr_y_logistic_feature):    # build model for each trait separately

            Logger.info('')
            Logger.info('build model for: {}'.format(str(y_feature)))

            X, y = self._prepare_raw_data_to_model(y_feature, relevant_X_columns)

            # whether to use cross validation or just train-test
            # X_train, X_test, y_train, y_test = self._split_data(X, y)
            # majority_ratio = max(round(sum(y) / len(y), 2), 1 - round(sum(y) / len(y), 2))
            data_size = X.shape[0]
            majority_ratio = 0

            if True:
                if self.classifier_type == 'xgb':
                    regr = XGBClassifier(
                        n_estimators=self.xgb_n_estimators,
                        max_depth=self.xgb_max_depth,
                        learning_rate=self.xgb_eta,
                        gamma=self.xgb_c,
                        subsample=self.xgb_subsample,
                        colsample_bytree=self.xgb_colsample_bytree
                    )
                elif self.classifier_type == 'lr':
                    self.penalty = random.choice(['l1', 'l2'])
                    regr = linear_model.LogisticRegression(
                        penalty=self.penalty,
                        C=self.xgb_c,
                        solver='liblinear'
                    )
                else:
                    raise ValueError('unknown classifier type - {}'.format(self.classifier_type))

                # implement cross validation
                acc_arr, auc_arr = self._cross_validation_implementation(regr, X, y, y_feature)
                acc_mean = round(sum(acc_arr)/len(acc_arr), 2)
                auc_mean = round(sum(auc_arr)/len(auc_arr), 2)

                # extract feature importance
                dict_importance = {}
                dict_param = self._log_parameters_order(dict_importance)

                Logger.info("{} results:".format(y_feature))
                Logger.info('feature importance (SelectKBest): {}'.format(dict_param))
                Logger.info('Accuracy: {}'.format(acc_mean, ))
                Logger.info('AUC: {}'.format(auc_mean, ))
                Logger.info('Accuracy list: {}'.format(acc_arr))
                Logger.info('AUC list: {}'.format(auc_arr))
                Logger.info(regr)

                # dict_param = dict(zip(k_feature, regr.feature_importances_))
            else:
                raise ValueError('unknown classifier type - {}'.format(self.classifier_type))

            # dict_param = self._log_parameters_order(dict_param)
            # dict_param = {}
            # insert current model to store later in a CSV
            self._prepare_model_result(
                y_feature, acc_mean, auc_mean, '', dict_param, data_size, majority_ratio, acc_arr, auc_arr, X
            )

    def _cross_validation_implementation(self, _regr, _X, _y, y_feature):
        acc_arr = list()
        auc_arr = list()

        # kfold = StratifiedKFold(n_splits=self.n_splits, random_state=None, shuffle=False)
        # for train_index, test_index in kfold.split(_X, _y):
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=42)
        skf.get_n_splits(_X, _y)
        for train_index, test_index in skf.split(_X, _y):
            X_train, X_test = _X.iloc[train_index], _X.iloc[test_index]
            y_train, y_test = _y.iloc[train_index], _y.iloc[test_index]

            # create textual features
            X_train, X_test, f_names = self._append_textual_features(X_train, X_test)

            # implement normalization
            if True:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            assert X_train.shape[1] == X_test.shape[1]
            assert X_train.shape[0] == y_train.shape[0]
            assert X_test.shape[0] == y_test.shape[0]

            # select K best
            if self.k_best_feature_flag:
                X_train, X_test, k_feature = self._select_k_best_feature(X_train, y_train, X_test, f_names)

            Logger.info('Classifier input train: {} test: {}'.format(X_train.shape, X_test.shape))
            _regr.fit(X_train, y_train)

            probas_ = _regr.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            cur_acc = round(_regr.score(X_test, y_test), 3)
            cur_auc = round(auc(fpr, tpr), 3)

            acc_arr.append(cur_acc)
            auc_arr.append(cur_auc)

        return acc_arr, auc_arr

    def _prepare_model_result(self, y_feature, test_acc, auc_score, train_acc, features, data_size,
                              majority_ratio, acc_arr, auc_arr, X):

        self.models_results.append({
            'method': bfi_config.predict_trait_configs['model_method'],
            'classifier': self.classifier_type,
            'CV_bool': 'False' if self.split_bool else 'True',
            'user_type': self.user_type,
            'l_limit': self.l_limit,
            'h_limit': self.h_limit,
            'threshold': self.threshold_purchase,
            'k_features': X.shape[1] if not self.k_best_feature_flag else self.k_best,
            'penalty': self.penalty if self.classifier_type == 'lr' else '',
            'xgb_gamma': self.xgb_c,
            'xgb_eta': self.xgb_eta,
            'xgb_max_depth': self.xgb_max_depth,
            'trait': y_feature.split('_')[0],
            'test_accuracy': test_acc,
            'auc': auc_score,
            'accuracy_k_fold': acc_arr,
            'auc_k_fold': auc_arr,
            'train_accuracy': train_acc,
            'data_size': data_size,
            'majority_ratio': majority_ratio,
            'features': features,
            'xgb_n_estimators': self.xgb_n_estimators,
            'xgb_subsample': self.xgb_subsample,
            'xgb_colsample_bytree': self.xgb_colsample_bytree
        })

    def _prepare_raw_data_to_model(self, y_feature, relevant_X_columns):
        """
        for a given target trait, prepare data set before insert into the model.
            1. slice data above/below percentile threshold
            2. normalize features
        """

        # Drop middle (Gap)
        cur_f = self.map_dict_percentile_group[y_feature]
        self.merge_df.to_csv('{}logistic_regression_merge_df.csv'.format(self.dir_analyze_name))

        h_df = self.merge_df.loc[self.merge_df[cur_f] >= self.h_limit]
        l_df = self.merge_df.loc[self.merge_df[cur_f] <= self.l_limit]

        Logger.info('H group amount: {}'.format(h_df.shape[0]))
        Logger.info('L group amount: {}'.format(l_df.shape[0]))

        frames = [l_df, h_df]
        self.raw_df = pd.concat(frames)

        self.raw_df = self.raw_df.fillna(0)

        # Drop Un-relevant columns

        relevant_X_columns = copy.deepcopy(self.lr_x_feature) + ['buyer_id']
        if y_feature in relevant_X_columns:
            relevant_X_columns.remove(y_feature)

        before_drop_columns = self.raw_df.shape[1]
        self.raw_df = self.raw_df[relevant_X_columns + [y_feature]]
        Logger.info('Removed un-relevant columns: {}-{}={}'.format(
            before_drop_columns, self.raw_df.shape[1], before_drop_columns-self.raw_df.shape[1]))

        # Drop constant features
        old_df_features = list(self.raw_df)
        self.raw_df = self.raw_df.loc[:, self.raw_df.apply(pd.Series.nunique) != 1]
        Logger.info('Removed constant features: {}'.format(list(set(self.raw_df) - set(old_df_features))))

        # Normalized features - not in use here
        # assert self.bool_normalize_features

        # if self.bool_normalize_features:
        #    self.raw_df = self.preprocessing_min_max(self.raw_df)

        # create X, y (not normalized)
        X = self.raw_df[list(set(relevant_X_columns) & set(list(self.raw_df)))]
        y = self.raw_df[y_feature]

        return X, y

    def _split_data(self, X, y):
        """
        split data-set into train and test sets according to split bool
        """
        Logger.info('all: class 0 ratio: {}'.format(str(round(sum(y) / len(y), 2))))

        if self.split_bool:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                stratify=y,
                test_size=self.test_fraction
            )
            Logger.info('train: class 0 ratio: {}'.format(str(round(sum(y_train) / len(y_train), 2))))
            Logger.info('test: class 0 ratio: {}'.format(str(round(sum(y_test) / len(y_test), 2))))
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None

        return X_train, X_test, y_train, y_test

    def _select_k_best_feature(self, X_train, y_train, X_test, f_names):
        """
        select K-best feature and transform feature set to k-feature.
        """
        if self.k_best == 'all' or self.k_best > X_train.shape[1]:
            Logger.info('changed K value to all because k valus is below number of features')
            self.k_best = 'all'

        before_cols = X_train.shape[1]
        k_model = SelectKBest(f_classif, k=self.k_best)
        k_model.fit(X_train, y_train)

        idxs_selected = k_model.get_support(indices=True)
        k_feature = list()

        for idx, cur_feature in enumerate(f_names):
            if idx in idxs_selected:
                k_feature.append(cur_feature)

        X_train = k_model.transform(X_train)
        X_test = k_model.transform(X_test)

        assert X_train.shape[1] == X_test.shape[1]

        Logger.info('Sample size: {}, feature before: {}, after: {}'.format(
            X_train.shape[0] + X_test.shape[0],
            before_cols,
            X_train.shape[1]
        ))
        Logger.info('K feature selected: {}, names: {}'.format(self.k_best, ', '.join(k_feature)))

        return X_train, X_test, k_feature

    def _log_parameters_order(self, dict_param):
        """

        :param dict_param: key=feature val=coefficients/importance
        :return:
        """
        d_view = [(round(v, 2), k) for k, v in dict_param.iteritems()]
        d_view.sort(reverse=True)

        Logger.info("")
        Logger.info("Model Parameters:")

        for v, k in d_view:
            if v != 0:
                Logger.info("{}: {}".format(k, round(v, 3)))
        Logger.info("")
        return d_view

    # add High/Low based percentile to traits
    def add_high_low_traits_column(self):

        self.merge_df['openness_group'] = str
        self.merge_df['conscientiousness_group'] = str
        self.merge_df['extraversion_group'] = str
        self.merge_df['agreeableness_group'] = str
        self.merge_df['neuroticism_group'] = str

        self.merge_df['openness_group'] = \
            np.where(self.merge_df['openness_percentile'] >= 0.5, 1, 0)

        self.merge_df['conscientiousness_group'] = \
            np.where(self.merge_df['conscientiousness_percentile'] >= 0.5, 1, 0)

        self.merge_df['extraversion_group'] = \
            np.where(self.merge_df['extraversion_percentile'] >= 0.5, 1, 0)

        self.merge_df['agreeableness_group'] = \
            np.where(self.merge_df['agreeableness_percentile'] >= 0.5, 1, 0)

        self.merge_df['neuroticism_group'] = \
            np.where(self.merge_df['neuroticism_percentile'] >= 0.5, 1, 0)

        self.merge_df.to_csv(self.dir_analyze_name + 'logistic_regression_df.csv')
        Logger.info('')
        Logger.info('add High/Low traits group')
        Logger.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'logistic_regression_df.csv')

    # normalized each column seperatly between zero to one
    def preprocessing_min_max(self, df):

        from sklearn import preprocessing
        norm_method = 'min_max'

        # remove constant column
        old_df_features = list(df)
        df = df.loc[:, df.apply(pd.Series.nunique) != 1]
        Logger.info('removed constant features: {}'.format(
            list(set(df) - set(old_df_features)))
        )

        if norm_method == 'min_max':
            normalized_df = (df - df.min()) / (df.max() - df.min())
        elif norm_method == 'mean_normalization':
            normalized_df = (df-df.mean())/df.std()
        else:
            raise ValueError('undefined norm method')

        # normalized_df.to_csv(self.dir_analyze_name + 'norm_df.csv')
        return normalized_df

    # save csv
    def save_data(self):
        self.df.reset_index(drop=True, inplace=True)
        save_file_name = self.file_name[:-4] + '_manipulate.csv'
        self.df.to_csv(save_file_name)
        return

    # reverse all relevant values
    def change_reverse_value(self):
        reverse_col = [2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43]
        for cur_rcol in reverse_col:
            start_str_cur = str(cur_rcol) + '.'
            filter_col = [col for col in self.participant_df if col.startswith(start_str_cur)][0]
            # print(filter_col)
            Logger.info('Change column values (reverse mode): ' + str(filter_col))
            self.participant_df[filter_col] = self.participant_df[filter_col].apply(lambda x: 6 - x)
        return

    def cal_participant_percentile_traits_values(self):
        # add empty columns

        self.merge_df["openness_percentile"] = np.nan
        self.merge_df["conscientiousness_percentile"] = np.nan
        self.merge_df["extraversion_percentile"] = np.nan
        self.merge_df["agreeableness_percentile"] = np.nan
        self.merge_df["neuroticism_percentile"] = np.nan
        self.merge_df["age_group"] = ''  # a (15-24), b (25-29), c(30-34), d(35-39), e(40-100)

        self.openness_score_list = self.merge_df['openness_trait'].tolist()
        self.conscientiousness_score_list = self.merge_df['conscientiousness_trait'].tolist()
        self.extraversion_score_list = self.merge_df['extraversion_trait'].tolist()
        self.agreeableness_score_list = self.merge_df['agreeableness_trait'].tolist()
        self.neuroticism_score_list = self.merge_df['neuroticism_trait'].tolist()

        # add percentile traits columns
        for (idx, row_participant) in self.merge_df.iterrows():
            self._cal_participant_traits_percentile_values(idx, row_participant)

    # after delete un valid participant
    def cal_all_participant_percentile_value(self):
        for (idx, row_participant) in self.participant_df.iterrows():
            Logger.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self._cal_participant_traits_percentile_values(idx, row_participant)

            # after calculate traits score+percentile extract only relevant features
        remain_feature_list = ['Full Name', 'Gender', 'eBay site user name', 'Age', 'openness_trait',
                               'conscientiousness_trait', 'extraversion_trait', 'agreeableness_trait',
                               'neuroticism_trait', 'openness_percentile', 'conscientiousness_percentile',
                               'extraversion_percentile', 'agreeableness_percentile', 'neuroticism_percentile',
                               'age_group']

        # self.merge_df = self.merge_df[remain_feature_list].copy()
        return

    # calculate average traits value
    def cal_participant_traits(self, row, cur_trait_list, ratio):
        trait_val = 0
        for cur_col in cur_trait_list:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.participant_df if col.startswith(start_str_cur)][0]   # find col name
            trait_val += row[filter_col]
        trait_val = float(trait_val)/float(len(cur_trait_list))     # mean of traits
        return trait_val

    def _eli5_explain_weights(self, clf, f_names, auc, target_name):
        """Explains top K features with the highest coefficient values, per class, using eli5"""

        # avoid overload on eli5 folder
        if auc < 0.75:
            return

        Logger.info('eli5_explain_weights:')

        dir_output = '/Users/gelad/Personality-based-commerce/results/BFI_results/classifiers_feature_importance/{}'.format(
            target_name)
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        if isinstance(clf, XGBClassifier):
            c_type = 'XGB'
            eli5_ew = explain_weights_xgboost(clf.get_booster(), feature_names=f_names, target_names=[target_name])

            wf_path = dir_output + '/{}_{}_threshold={}_k={}_type={}_n_estimator={}_depth={}_eta={}_c={}_sub={}_col={}_{}.html'.format(
                auc,
                c_type,
                self.threshold_purchase,
                self.k_best,
                self.user_type,
                self.xgb_n_estimators,
                self.xgb_max_depth,
                self.xgb_eta,
                self.xgb_c,
                self.xgb_subsample,
                self.xgb_colsample_bytree,
                self.cur_time
            )

            # eli5_ew = show_weights(clf.get_booster(), feature_names=f_names)
        elif isinstance(clf, linear_model.LogisticRegression):
            c_type = 'LR'
            eli5_ew = explain_linear_classifier_weights(
                clf,
                feature_names=f_names,
                target_names=['Low', 'High'],
                top=100
            )

            # self.penalty
            # write explanation html to file
            wf_path = dir_output + '/{}_{}_threshold={}_k={}_type={}_penalty={}_c={}_{}.html'.format(
                auc,
                c_type,
                self.threshold_purchase,
                self.k_best,
                self.user_type,
                self.penalty,
                self.xgb_c,
                self.cur_time
            )

        else:
            # Logger.info('unsupported clf type')
            raise ValueError("unsupported clf type")

        eli5_fh = formatters.format_as_html(eli5_ew)

        # write explanation html to file
        prefix_to_html = ''

        lines_final = prefix_to_html + eli5_fh
        # lines_final = prefix_to_html + eli5_fh.encode('utf8', 'replace')

        Logger.info("writing weight explanation to file {}".format(wf_path))
        with open(wf_path, 'w') as wf:
            wf.writelines(lines_final)


def main():
    pass

if __name__ == '__main__':
    raise SystemExit('not in use - please run using Wrapper_build_feature_dataset')

