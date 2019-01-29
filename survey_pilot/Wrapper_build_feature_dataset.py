from __future__ import print_function
from build_feature_dataset import CalculateScore        # class which extract data set from input files
from utils.logger import Logger
from utils.utils import load_gensim_kv, load_gensim_kv_fasttext
from time import gmtime, strftime
import os
import pandas as pd
import sys
# from utils.single_embedding import MeanEmbeddingVectorizer

reload(sys)
sys.setdefaultencoding('utf8')

from config import bfi_config

participant_file = bfi_config.predict_trait_configs['participant_file']
item_aspects_file = bfi_config.predict_trait_configs['item_aspects_file']
purchase_history_file = bfi_config.predict_trait_configs['purchase_history_file']
valid_users_file = bfi_config.predict_trait_configs['valid_users_file']
dir_analyze_name = bfi_config.predict_trait_configs['dir_analyze_name']
dir_logistic_results = bfi_config.predict_trait_configs['dir_logistic_results']
dict_feature_flag = bfi_config.predict_trait_configs['dict_feature_flag']

predefined_data_set_flag = bfi_config.predict_trait_configs['predefined_data_set_flag']     # compute data_set or load
predefined_data_set_path = bfi_config.predict_trait_configs['predefined_data_set_path']

model_method = bfi_config.predict_trait_configs['model_method']
classifier_type = bfi_config.predict_trait_configs['classifier_type']
split_bool = bfi_config.predict_trait_configs['split_bool']

user_type = bfi_config.predict_trait_configs['user_type']
l_limit = bfi_config.predict_trait_configs['l_limit']
h_limit = bfi_config.predict_trait_configs['h_limit']

k_best_feature_flag = bfi_config.predict_trait_configs['k_best_feature_flag']
k_best_list = bfi_config.predict_trait_configs['k_best_list']
threshold_list = bfi_config.predict_trait_configs['threshold_list']
penalty = bfi_config.predict_trait_configs['penalty']
C = bfi_config.predict_trait_configs['C']

xgb_c = bfi_config.predict_trait_configs['xgb_c']
xgb_eta = bfi_config.predict_trait_configs['xgb_eta']
xgb_max_depth = bfi_config.predict_trait_configs['xgb_max_depth']

bool_slice_gap_percentile = bfi_config.predict_trait_configs['bool_slice_gap_percentile']


embedding_dim = bfi_config.predict_trait_configs['embedding_dim']
embedding_limit = bfi_config.predict_trait_configs['embedding_limit']
embedding_type = bfi_config.predict_trait_configs['embedding_type']


class Wrapper:

    def __init__(self):

        if dict_feature_flag['title_feature_flag'] or dict_feature_flag['descriptions_feature_flag'] or dict_feature_flag['color_feature_flag']:
            print('Loading embeddings...    {}'.format(embedding_type))
            if embedding_type == 'ft_amazon':
                path_embd = '/Users/gelad/Personality-based-commerce/data/word_embedding/embeddings.vec'
                self.kv = load_gensim_kv(path_embd, vector_size=embedding_dim, limit=embedding_limit)
                assert len(self.kv.vocab) == embedding_limit

            elif embedding_type == 'glove':
                path_embd = '/Users/gelad/Personality-based-commerce/data/word_embedding/glove.6B.{}d.txt'.format(
                    embedding_dim)
                self.kv = load_gensim_kv(path_embd, vector_size=embedding_dim, limit=embedding_limit)
                assert len(self.kv.vocab) == embedding_limit

            elif embedding_type == 'fasttext':
                path_embd = '/Users/gelad/Personality-based-commerce/data/word_embedding/cc.en.300.bin'
                self.kv = load_gensim_kv_fasttext(path_embd, vector_size=embedding_dim, limit=embedding_limit)

            else:
                raise ValueError('unknown embedding type')

            # print('Loading embeddings...    {}'.format(path_embd))
            # self.kv = load_gensim_kv(path_embd, vector_size=embedding_dim, limit=embedding_limit)

            print('Loaded embedding!        dim: {}, limit: {}'.format(embedding_dim, embedding_limit))

        else:
            print('Embeddings did not loaded - flage set to False')
            self.kv = None

        # file arguments
        self.participant_file = participant_file
        self.item_aspects_file = item_aspects_file
        self.purchase_history_file = purchase_history_file
        self.valid_users_file = valid_users_file
        self.dir_analyze_name = dir_analyze_name
        self.dir_logistic_results = dir_logistic_results

        self.predefined_data_set_flag = predefined_data_set_flag
        self.predefined_data_set_path = predefined_data_set_path

        self.log_file_name = None
        self.num_experiments = None

        self.user_type = user_type                  # which user to keep in model 'all'/'cf'/'ebay-tech'
        self.model_method = model_method            # 'linear'/'logistic'
        self.classifier_type = classifier_type      # 'xgb' ...
        self.split_bool = split_bool                # whether to use cross-validation or just train-test

        self.k_best_feature_flag = k_best_feature_flag
        self.k_best_list = k_best_list
        self.threshold_list = threshold_list
        self.penalty = penalty
        self.bool_slice_gap_percentile = bool_slice_gap_percentile
        self.h_limit = h_limit
        self.l_limit = l_limit

        self.threshold_purchase = 1
        self.bool_normalize_features = True
        self.C = C
        self.cur_penalty = None

        self.xgb_c = xgb_c
        self.xgb_eta = xgb_eta
        self.xgb_max_depth = xgb_max_depth

        self.split_test = False
        self.normalize_traits = True    # normalize traits to 0-1 (divided by 5)

        # bool values for which feature will be in the model (before sliced by in the selectKbest)
        self.time_purchase_ratio_feature_flag = dict_feature_flag['time_purchase_ratio_feature_flag']
        self.time_purchase_meta_feature_flag = dict_feature_flag['time_purchase_meta_feature_flag']
        self.vertical_ratio_feature_flag = dict_feature_flag['vertical_ratio_feature_flag']
        self.meta_category_feature_flag = dict_feature_flag['meta_category_feature_flag']
        self.purchase_percentile_feature_flag = dict_feature_flag['purchase_percentile_feature_flag']
        self.user_meta_feature_flag = dict_feature_flag['user_meta_feature_flag']
        self.aspect_feature_flag = dict_feature_flag['aspect_feature_flag']
        self.color_feature_flag = dict_feature_flag['color_feature_flag']

        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.plot_directory = self.dir_logistic_results + str(self.model_method) + '_results/'
        if self.model_method == 'logistic':
            self.plot_directory += 'gap_' + str(self.h_limit) + '_' + str(self.l_limit) + '_time_' + str(self.cur_time) + '/'
        elif self.model_method == 'linear':
            self.plot_directory += 'time_' + str(self.cur_time) + '/'
        else:
            raise ValueError('unknown method type')

        an = self
        self.input_attr_dict = vars(an)  # save attribute to show them later

        if not os.path.exists(self.plot_directory):
            os.makedirs(self.plot_directory)

        if not os.path.exists(self.dir_analyze_name):
            os.makedirs(self.dir_analyze_name)

        """self.result_df = pd.DataFrame(columns=[
            'method', 'classifier', 'CV_bool', 'user_type', 'l_limit', 'h_limit', 'regularization_type', 'C',
            'threshold', 'k_features', 'xgb_c', 'xgb_eta', 'xgb_max_depth', 'trait', 'test_accuracy', 'auc',
            'accuracy_k_fold', 'auc_k_fold', 'train_accuracy', 'test_size', 'train_size', 'majority_ratio', 'features'
        ])"""

        self.result_df = pd.DataFrame(columns=[
            'method', 'classifier', 'CV_bool', 'user_type', 'l_limit', 'h_limit',
            'threshold', 'k_features', 'k_flag', 'penalty', 'xgb_gamma', 'xgb_eta', 'xgb_max_depth', 'trait', 'test_accuracy', 'auc',
            'accuracy_k_fold', 'auc_k_fold', 'train_accuracy', 'data_size', 'majority_ratio', 'features',
            'xgb_n_estimators', 'xgb_subsample', 'xgb_colsample_bytree', 'emb_dim', 'emb_limit', 'emb_type', 'vec_type', 'vec_max_feature', 'vec_missing_val', 'vec_min_df', 'vec_max_df'
        ])

        self.max_score = {
            "openness": {'i': -1, 'score': 0.0},
            "agreeableness": {'i': -1, 'score': 0.0},
            "conscientiousness": {'i': -1, 'score': 0.0},
            "extraversion": {'i': -1, 'score': 0.0},
            "neuroticism": {'i': -1, 'score': 0.0},
        }

        self.i = int
        self.k_best = int

    ############################# validation functions #############################

    # build log object
    def init_debug_log(self):
        file_prefix = 'wrapper_build_feature_data_set'
        self.log_file_name = '../log/BFI/{}_{}.log'.format(file_prefix, self.cur_time)
        Logger.set_handlers('Wrapper build feature', self.log_file_name, level='info')

    # check if input data is valid
    def check_input(self):

        if self.user_type not in ['all', 'cf', 'ebay-tech']:
            raise ValueError('unknown user_type')

        if self.model_method not in ['linear', 'logistic']:
            raise ValueError('unknown model_method')

        if len(self.k_best_list) < 1:
            raise ValueError('empty k_best_list')

        if len(self.threshold_list) < 1:
            raise ValueError('empty threshold_list')

        if len(self.penalty) < 1:
            raise ValueError('empty penalty')

        if self.h_limit < 0 or self.h_limit > 1:
            raise ValueError('h_limit must be a float between 0 to 1')

        if self.l_limit < 0 or self.l_limit > 1:
            raise ValueError('l_limit must be a float between 0 to 1')

        if self.l_limit >= self.h_limit:
            raise ValueError('l_limit must be smaller than h_limit')

        if self.classifier_type == 'xgb' and not self.split_bool:
            raise ValueError('currently we do not support cross validation for XGB model')

        if bfi_config.feature_data_set['categ_threshold'] not in [0, 250, 500]:
            raise ValueError('categ purchase threshold not in list [0, 250, 500]')

    # log class arguments
    def log_attribute_input(self):

        Logger.info('')
        Logger.info('Class arguments')
        import collections
        od = collections.OrderedDict(sorted(self.input_attr_dict.items()))
        for attr, attr_value in od.iteritems():
            if type(attr_value) is list and len(attr_value)==0:
                continue
            Logger.info('Attribute: ' + str(attr) + ', Value: ' + str(attr_value))
        Logger.info('')

    ############################# main functions #############################
    '''
    table of contents (chronological order)
    1. run models - router correspond to model received 
    2. wrapper_experiments_logistic/linear - run configurations 
    3. run_experiments - for a specific configuration call inner class
        3.1 build data set (mainly using threshold + features defined)
        3.2 run model and store results
    '''
    # which model ('linear'/'logistic')
    def run_models(self):

        self.check_input()          # check if input argument are valid
        self.log_attribute_input()  # log arguments for all model

        if self.model_method == 'logistic':
            self.wrapper_experiments_logistic()
        else:
            raise ValueError('unknown model methods: {}'.format(self.model_method))

    # run if we choose logistic method
    # run all possible configurations given inputs
    def wrapper_experiments_logistic(self):

        self.num_experiments = len(self.penalty)*len(self.k_best_list)*len(self.xgb_c)*len(self.xgb_eta)*\
                               len(self.xgb_max_depth)*len(self.threshold_list)*len(bfi_config.feature_data_set['lr_y_logistic_feature'])  # number of traits

        for cur_penalty in self.penalty:
            for k_best in self.k_best_list:
                for xgb_c in self.xgb_c:
                    for xgb_eta in self.xgb_eta:
                        for xgb_max_depth in self.xgb_max_depth:
                            for threshold_purchase in self.threshold_list:

                                self.threshold_purchase = threshold_purchase
                                self.k_best = k_best
                                self.cur_penalty = cur_penalty      # TODO change

                                Logger.info('')
                                Logger.info('')
                                Logger.info('######################### run new configuration ########################')
                                Logger.info('')
                                Logger.info('Current configuration: Penalty: {}, Threshold: {}, k_best: {}'.format(
                                    cur_penalty,
                                    threshold_purchase,
                                    k_best
                                ))

                                calculate_obj = self.run_experiments(xgb_c, xgb_eta, xgb_max_depth)     # run configuration

                                # insert current model result
                                self._store_data_df(calculate_obj.models_results)

        self._save_result_df()
        self._save_to_ablation_csv()

    # run experiments for a giving arguments
    def run_experiments(self, xgb_c, xgb_eta, xgb_max_depth):
        calculate_obj = CalculateScore(participant_file, item_aspects_file, purchase_history_file, valid_users_file,
                                       dir_analyze_name, self.threshold_purchase, self.bool_slice_gap_percentile,
                                       self.bool_normalize_features, self.C, self.cur_penalty,
                                       self.time_purchase_ratio_feature_flag, self.time_purchase_meta_feature_flag,
                                       self.vertical_ratio_feature_flag, self.meta_category_feature_flag, self.purchase_percentile_feature_flag,
                                       self.user_meta_feature_flag, self.aspect_feature_flag, self.color_feature_flag, self.h_limit,
                                       self.l_limit, self.k_best, self.plot_directory, self.user_type,
                                       self.normalize_traits, self.classifier_type, self.split_bool, xgb_c, xgb_eta,
                                       xgb_max_depth, self.dir_logistic_results, self.cur_time, self.k_best_feature_flag, self.kv)

        # create data set
        if not self.predefined_data_set_flag:
            Logger.info('Start creating data set')

            calculate_obj.load_clean_csv_results()                  # load data set
            calculate_obj.clean_df()                                # clean df - e.g. remain valid users only

            calculate_obj.insert_gender_feature()                   # add gender feature
            calculate_obj.remove_except_cf()                        # remove not CF participants
            calculate_obj.extract_user_purchase_connection()        # insert purchase and vertical type to model
            calculate_obj.insert_meta_category()                    # Add eBay meta categories features
            # calculate_obj.insert_titles_features_count()                  # add titles n-grams features
            calculate_obj.insert_color_features()
            calculate_obj.insert_textual_features_embedding('descriptions')  # add titles n-grams features
            calculate_obj.insert_textual_features_embedding('titles')                  # add titles n-grams features
            calculate_obj.insert_textual_features_embedding('colors')  # add titles n-grams features

            calculate_obj.insert_descriptions_features()            # add descriptions n-grams features

            calculate_obj.extract_item_aspect()                     # add features of dominant item aspect

            calculate_obj.normalize_personality_trait()                 # normalize trait to 0-1 scale (div by 5)

            # important!! after cut users by threshold
            calculate_obj.cal_participant_percentile_traits_values()    # calculate average traits and percentile value

            calculate_obj.insert_money_feature()                        # add feature contain money issue
            calculate_obj.insert_time_feature()                         # add time purchase feature
            calculate_obj.save_predefined_data_set()

            # moved here after we add textual features both titles and descriptions
            calculate_obj.create_feature_list()  # create x_feature

        # load pre-trained data set
        else:
            Logger.info('load pre-trained data set')
            calculate_obj.create_feature_list()
            calculate_obj.load_predefined_data_set(self.predefined_data_set_path)

        # run logistics models (XGB/LR) - predict whether trait is H or L
        calculate_obj.calculate_logistic_regression()

        return calculate_obj

    ############################# store in a CSV #############################

    def _store_data_df(self, model_results_array):
        """
        insert model result for a given configuration
        """
        cur_configs_five = list()
        for row in model_results_array:
            config_result_dict = {
                'method': row['method'],
                'classifier': row['classifier'],
                'CV_bool': row['CV_bool'],
                'user_type': row['user_type'],
                'l_limit': row['l_limit'],
                'h_limit': row['h_limit'],
                # 'regularization_type': row['regularization_type'],
                # 'C': row['C'],
                'threshold': row['threshold'],
                'k_features': row['k_features'],
                'k_flag': bfi_config.predict_trait_configs['k_best_feature_flag'],
                'penalty': row['penalty'],
                'xgb_gamma': row['xgb_gamma'],
                'xgb_eta': row['xgb_eta'],
                'xgb_max_depth': row['xgb_max_depth'],
                'trait': row['trait'],
                'test_accuracy': row['test_accuracy'],
                'auc': row['auc'],
                'accuracy_k_fold': row['accuracy_k_fold'],
                'auc_k_fold': row['auc_k_fold'],
                'train_accuracy': row['train_accuracy'],
                'data_size': row['data_size'],
                'majority_ratio': row['majority_ratio'],
                'features': row['features'],
                'xgb_n_estimators': row['xgb_n_estimators'],
                'xgb_subsample': row['xgb_subsample'],
                'xgb_colsample_bytree': row['xgb_colsample_bytree'],
                'emb_dim': bfi_config.predict_trait_configs['embedding_dim'],
                'emb_limit': bfi_config.predict_trait_configs['embedding_limit'],
                'emb_type': bfi_config.predict_trait_configs['embedding_type'],
                'vec_type': bfi_config.predict_trait_configs['dict_vec']['vec_type'],
                'vec_max_feature': bfi_config.predict_trait_configs['dict_vec']['max_features'],
                'vec_missing_val': bfi_config.predict_trait_configs['dict_vec']['missing_val'],
                'vec_min_df': bfi_config.predict_trait_configs['dict_vec']['min_df'],
                'vec_max_df': bfi_config.predict_trait_configs['dict_vec']['max_df']
            }

            cur_configs_five.append(config_result_dict)
            self.result_df = self.result_df.append(config_result_dict, ignore_index=True)

            Logger.info('insert model number into result df: {}/{}, {}%'.format(
                self.result_df.shape[0],
                self.num_experiments,
                round(float(self.result_df.shape[0])/self.num_experiments, 2)*100
            ))

        """ insert one row """
        result_df_path = os.path.join(self.dir_logistic_results, 'intermediate_models')
        result_df_path = os.path.join(result_df_path, '{}.csv'.format(self.cur_time))
        """if os.path.isfile(result_df_path):
            intermediate_df = pd.read_csv(result_df_path)
        else:
            intermediate_df = pd.DataFrame()

        for one_run in cur_configs_five:
            intermediate_df = intermediate_df.append(one_run, ignore_index=True)"""

        self.result_df.to_csv(result_df_path, index=False)
        Logger.info('update intermediate df: {} path :{}'.format(self.result_df.shape[0], result_df_path))

    def _save_result_df(self):
        """
        save all model results
        :return:
        """
        result_df_path = os.path.join(self.dir_logistic_results, 'compare_models')
        if not os.path.exists(result_df_path):
            os.makedirs(result_df_path)
        list_e = self.result_df.loc[self.result_df['trait'] == 'extraversion']['auc']
        list_o = self.result_df.loc[self.result_df['trait'] == 'openness']['auc']
        list_a = self.result_df.loc[self.result_df['trait'] == 'agreeableness']['auc']
        list_n = self.result_df.loc[self.result_df['trait'] == 'neuroticism']['auc']
        list_c = self.result_df.loc[self.result_df['trait'] == 'conscientiousness']['auc']

        e = round(max(list_e), 2) if list_e.tolist() else 0
        o = round(max(list_o), 2) if list_o.tolist() else 0
        a = round(max(list_a), 2) if list_a.tolist() else 0
        n = round(max(list_n), 2) if list_n.tolist() else 0
        c = round(max(list_c), 2) if list_c.tolist() else 0
        """
        e = round(max(self.result_df.loc[self.result_df['trait'] == 'extraversion']['auc']), 2)
        o = round(max(self.result_df.loc[self.result_df['trait'] == 'openness']['auc']), 2)
        a = round(max(self.result_df.loc[self.result_df['trait'] == 'agreeableness']['auc']), 2)
        n = round(max(self.result_df.loc[self.result_df['trait'] == 'neuroticism']['auc']), 2)
        c = round(max(self.result_df.loc[self.result_df['trait'] == 'conscientiousness']['auc']), 2)
        """
        best_acc = max(o, c, e, a, n)
        num_splits = bfi_config.predict_trait_configs['num_splits']
        title_features = bfi_config.predict_trait_configs['dict_feature_flag']['title_feature_flag']
        prefix_name = '{}_e={}_o={}_a={}_c={}_n={}_cnt={}_clf={}_user={}_split={}_title={}_h={}_l={}'.format(
            best_acc, e, o, a, c, n,
            self.result_df.shape[0], self.classifier_type, self.user_type, num_splits, title_features,
            self.h_limit, self.l_limit
        )

        result_df_path = os.path.join(result_df_path, '{}_{}.csv'.format(prefix_name, self.cur_time))
        self.result_df.to_csv(result_df_path, index=False)
        Logger.info('save result model: {}'.format(result_df_path))

    def _save_to_ablation_csv(self):
        """
        add a row to ablation csv file
        :return:
        """
        try:
            import os
            result_df_path = os.path.join(self.dir_logistic_results, 'ablation_test')
            if not os.path.exists(result_df_path):
                os.makedirs(result_df_path)
            result_df_path = os.path.join(result_df_path, '{}.csv'.format('ablation'))

            list_e = self.result_df.loc[self.result_df['trait'] == 'extraversion']['auc']
            list_o = self.result_df.loc[self.result_df['trait'] == 'openness']['auc']
            list_a = self.result_df.loc[self.result_df['trait'] == 'agreeableness']['auc']
            list_n = self.result_df.loc[self.result_df['trait'] == 'neuroticism']['auc']
            list_c = self.result_df.loc[self.result_df['trait'] == 'conscientiousness']['auc']

            e = round(max(list_e), 2) if list_e.tolist() else 0
            o = round(max(list_o), 2) if list_o.tolist() else 0
            a = round(max(list_a), 2) if list_a.tolist() else 0
            n = round(max(list_n), 2) if list_n.tolist() else 0
            c = round(max(list_c), 2) if list_c.tolist() else 0
            """
            changed because we want to examine sometimes only few traits (and the series will be empty - exception due to max())
            o = round(max(self.result_df.loc[self.result_df['trait'] == 'openness']['auc']), 2)
            c = round(max(self.result_df.loc[self.result_df['trait'] == 'conscientiousness']['auc']), 2)
            e = round(max(self.result_df.loc[self.result_df['trait'] == 'extraversion']['auc']), 2)
            a = round(max(self.result_df.loc[self.result_df['trait'] == 'agreeableness']['auc']), 2)
            n = round(max(self.result_df.loc[self.result_df['trait'] == 'neuroticism']['auc']), 2)
            """

            import csv
            dict_val = {
                    'time_purchase_ratio': self.time_purchase_ratio_feature_flag,
                    'time_purchase_meta': self.time_purchase_meta_feature_flag,
                    'vertical': self.vertical_ratio_feature_flag,
                    'meta_categories': self.meta_category_feature_flag,
                    'purchase_percentile': self.purchase_percentile_feature_flag,
                    'user_meta': self.user_meta_feature_flag,
                    'item_aspects': self.aspect_feature_flag,
                    'color': self.color_feature_flag,
                    'text_title': bfi_config.predict_trait_configs['dict_feature_flag']['title_feature_flag'],
                    'text_desc': bfi_config.predict_trait_configs['dict_feature_flag'][
                    'descriptions_feature_flag'],
                    'o': o,
                    'c': c,
                    'e': e,
                    'a': a,
                    'n': n,
                    'time': self.cur_time,
                    'classifier': self.classifier_type,
                    'n_splits': bfi_config.predict_trait_configs['num_splits'],
                    'l_limit': self.l_limit,
                    'h_limit': self.h_limit,
                    'thresholds': self.threshold_list,
                    'user_type': self.user_type,
                    'k_flag': self.k_best_feature_flag,
                    'vectorizer_type': bfi_config.predict_trait_configs['dict_vec']['vec_type'],
                    'emb_type': bfi_config.predict_trait_configs['embedding_type'],
                    'emb_dim': bfi_config.predict_trait_configs['embedding_dim'],
                    'vectorizer_max_feature': bfi_config.predict_trait_configs['dict_vec']['max_features'],
                    'vec_min_df': bfi_config.predict_trait_configs['dict_vec']['min_df'],
                    'vec_max_df': bfi_config.predict_trait_configs['dict_vec']['max_df'],
                    'vec_missing': bfi_config.predict_trait_configs['dict_vec']['missing_val'],
            }
            d_ablation = pd.read_csv(result_df_path)

            # os.rename(result_df_path, '{}_{}.csv'.format(result_df_path[:-4], self.cur_time))
            backlog_path = '{}/ablation_backlog/{}.csv'.format(result_df_path[:-12], self.cur_time)
            os.rename(result_df_path, backlog_path)
            Logger.info('store last ablation to backlog: {}'.format(backlog_path))

            d_ablation = d_ablation.append(dict_val, ignore_index=True)
            d_ablation.to_csv(result_df_path, index=False)
            Logger.info('update ablation df: {}'.format(d_ablation.shape))

            """fieldnames = ['time_purchase_ratio',
                          'time_purchase_meta',
                          'vertical',
                          'meta_categories',
                          'purchase_percentile',
                          'user_meta',
                          'item_aspects',
                          'text_title',
                          'text_desc',
                          'o',
                          'c',
                          'e',
                          'a',
                          'n',
                          'time',
                          'classifier',
                          'n_splits',
                          'l_limit',
                          'h_limit']
                          """
            Logger.info('group bool flags:')
            Logger.info(dict_feature_flag)

        except Exception:
            Logger.info('Exception during insertion to ablation test')


def main():

    wrapper_obj = Wrapper()
    wrapper_obj.init_debug_log()        # init debug once - log file
    wrapper_obj.run_models()


if __name__ == '__main__':
    main()
