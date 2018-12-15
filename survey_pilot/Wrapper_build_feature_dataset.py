from __future__ import print_function
from build_feature_dataset import CalculateScore        # class which extract data set from input files
from utils.logger import Logger
from time import gmtime, strftime
import os
import pandas as pd

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


class Wrapper:

    def __init__(self):

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
        self.cur_penalty = 'l2'

        self.xgb_c = xgb_c
        self.xgb_eta = xgb_eta
        self.xgb_max_depth = xgb_max_depth

        self.split_test = False
        self.normalize_traits = True    # normalize traits to 0-1 (divided by 5)

        # bool values for which feature will be in the model (before sliced by in the selectKbest)
        self.time_purchase_ratio_feature_flag = dict_feature_flag['time_purchase_ratio_feature_flag']
        self.time_purchase_meta_feature_flag = dict_feature_flag['time_purchase_meta_feature_flag']
        self.vertical_ratio_feature_flag = dict_feature_flag['vertical_ratio_feature_flag']
        self.purchase_percentile_feature_flag = dict_feature_flag['purchase_percentile_feature_flag']
        self.user_meta_feature_flag = dict_feature_flag['user_meta_feature_flag']
        self.aspect_feature_flag = dict_feature_flag['aspect_feature_flag']

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

        # "regular" accuracy results
        self.openness_score_list = list()
        self.conscientiousness_score_list = list()
        self.extraversion_score_list = list()
        self.agreeableness_score_list = list()
        self.neuroticism_score_list = list()

        # CV results
        self.openness_cv_score_list = list()
        self.conscientiousness_cv_score_list = list()
        self.extraversion_cv_score_list = list()
        self.agreeableness_cv_score_list = list()
        self.neuroticism_cv_score_list = list()

        # ROC results
        self.openness_score_roc_list = list()
        self.conscientiousness_score_roc_list = list()
        self.extraversion_score_roc_list = list()
        self.agreeableness_score_roc_list = list()
        self.neuroticism_score_roc_list = list()

        """self.result_df = pd.DataFrame(columns=[
            'method', 'classifier', 'CV_bool', 'user_type', 'l_limit', 'h_limit', 'regularization_type', 'C',
            'threshold', 'k_features', 'xgb_c', 'xgb_eta', 'xgb_max_depth', 'trait', 'test_accuracy', 'auc',
            'accuracy_k_fold', 'auc_k_fold', 'train_accuracy', 'test_size', 'train_size', 'majority_ratio', 'features'
        ])"""

        self.result_df = pd.DataFrame(columns=[
            'method', 'classifier', 'CV_bool', 'user_type', 'l_limit', 'h_limit',
            'threshold', 'k_features', 'xgb_gamma', 'xgb_eta', 'xgb_max_depth', 'trait', 'test_accuracy', 'auc',
            'accuracy_k_fold', 'auc_k_fold', 'train_accuracy', 'data_size', 'majority_ratio', 'features',
            'xgb_n_estimators', 'xgb_subsample', 'xgb_colsample_bytree'
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

        return

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

        if self.model_method == 'linear':
            raise ValueError('Currently support only logistic classifiers')
            self.wrapper_experiments_linear()

        elif self.model_method == 'logistic':
            self.wrapper_experiments_logistic()
        return

    # run if we choose logistic method
    # run all possible configurations given inputs
    def wrapper_experiments_logistic(self):

        self.num_experiments = len(self.penalty)*len(self.k_best_list)*len(self.xgb_c)*len(self.xgb_eta)*\
                               len(self.xgb_max_depth)*len(self.threshold_list)*5     # number of traits

        for cur_penalty in self.penalty:
            for k_best in self.k_best_list:
                for xgb_c in self.xgb_c:
                    for xgb_eta in self.xgb_eta:
                        for xgb_max_depth in self.xgb_max_depth:

                            # reset evaluation list.
                            # list target is to compare between configurations with same K_best and penalty
                            self.openness_score_list = list()
                            self.conscientiousness_score_list = list()
                            self.extraversion_score_list = list()
                            self.agreeableness_score_list = list()
                            self.neuroticism_score_list = list()

                            self.openness_score_roc_list = list()
                            self.conscientiousness_score_roc_list = list()
                            self.extraversion_score_roc_list = list()
                            self.agreeableness_score_roc_list = list()
                            self.neuroticism_score_roc_list = list()

                            self.openness_cv_score_list = list()
                            self.conscientiousness_cv_score_list = list()
                            self.extraversion_cv_score_list = list()
                            self.agreeableness_cv_score_list = list()
                            self.neuroticism_cv_score_list = list()

                            for threshold_purchase in self.threshold_list:

                                self.threshold_purchase = threshold_purchase
                                self.k_best = k_best
                                self.cur_penalty = cur_penalty      # TODO change

                                Logger.info('')
                                Logger.info('')
                                Logger.info('############################# run new configuration #############################')
                                Logger.info('')
                                Logger.info(
                                    'Current configuration: Penalty: ' + str(cur_penalty) + ', Threshold: ' + str(threshold_purchase) + ', k_best: ' + str(k_best)
                                )

                                calculate_obj = self.run_experiments(xgb_c, xgb_eta, xgb_max_depth)     # run configuration

                                # insert current model result
                                self._store_data_df(calculate_obj.models_results)

                                # store result correspond to whether we split data or not
                                if self.split_test:
                                    # test score
                                    self.openness_score_list.append(calculate_obj.logistic_regression_accuracy['openness'])
                                    self.conscientiousness_score_list.append(calculate_obj.logistic_regression_accuracy['conscientiousness'])
                                    self.extraversion_score_list.append(calculate_obj.logistic_regression_accuracy['extraversion'])
                                    self.agreeableness_score_list.append(calculate_obj.logistic_regression_accuracy['agreeableness'])
                                    self.neuroticism_score_list.append(calculate_obj.logistic_regression_accuracy['neuroticism'])

                                    # roc score
                                    self.openness_score_roc_list.append(calculate_obj.logistic_regression_roc['openness'])
                                    self.conscientiousness_score_roc_list.append(calculate_obj.logistic_regression_roc['conscientiousness'])
                                    self.extraversion_score_roc_list.append(calculate_obj.logistic_regression_roc['extraversion'])
                                    self.agreeableness_score_roc_list.append(calculate_obj.logistic_regression_roc['agreeableness'])
                                    self.neuroticism_score_roc_list.append(calculate_obj.logistic_regression_roc['neuroticism'])

                                # CV score
                                self.openness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['openness'])
                                self.conscientiousness_cv_score_list.append(
                                    calculate_obj.logistic_regression_accuracy_cv['conscientiousness'])
                                self.extraversion_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['extraversion'])
                                self.agreeableness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['agreeableness'])
                                self.neuroticism_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['neuroticism'])

                # plot results
                # if we split test+train we present test score + ROC curve
                # else we only present CV score without TODO ROC curve
                """
                if self.split_test:
                    self.plot_traits_accuracy_versus_threshold(cur_penalty, k_best)
                    self.plot_traits_roc_versus_threshold(cur_penalty, k_best)
                else:
                    self.plot_traits_accuracy_versus_threshold_CV(cur_penalty, k_best)
                """
        self._save_result_df()

    def wrapper_experiments_linear(self):

        self.k_best_list = [15, 12, 8, 5]
        self.threshold_list = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        penalty = ['l1', 'l2']

        # for cur_C in C:
        for cur_penalty in penalty:

            Logger.info('Penalty: ' + str(cur_penalty))
            for k_best in self.k_best_list:

                self.openness_score_mae_list = list()
                self.conscientiousness_score_mae_list = list()
                self.extraversion_score_mae_list = list()
                self.agreeableness_score_mae_list = list()
                self.neuroticism_score_mae_list = list()

                self.openness_score_pearson_list = list()
                self.conscientiousness_score_pearson_list = list()
                self.extraversion_score_pearson_list = list()
                self.agreeableness_score_pearson_list = list()
                self.neuroticism_score_pearson_list = list()

                '''self.openness_cv_score_list = list()
                self.conscientiousness_cv_score_list = list()
                self.extraversion_cv_score_list = list()
                self.agreeableness_cv_score_list = list()
                self.neuroticism_cv_score_list = list()'''

                for threshold_purchase in self.threshold_list:
                    self.threshold_purchase = threshold_purchase
                    self.k_best = k_best
                    self.cur_penalty = cur_penalty
                    Logger.info('Penalty: ' + str(cur_penalty) + ', Threshold: ' + str(threshold_purchase))

                    calculate_obj = self.run_experiments()

                    # cur_key = 'C_' + str(cur_C) + '_Penalty_' + str(cur_penalty) + '_Threshold_' + str(threshold_purchase)
                    cur_key = '_Penalty_' + str(cur_penalty) + '_Threshold_' + str(threshold_purchase)


                    # mae score
                    self.openness_score_mae_list.append(calculate_obj.linear_regression_mae['openness'])
                    self.conscientiousness_score_mae_list.append(calculate_obj.linear_regression_mae['conscientiousness'])
                    self.extraversion_score_mae_list.append(calculate_obj.linear_regression_mae['extraversion'])
                    self.agreeableness_score_mae_list.append(calculate_obj.linear_regression_mae['agreeableness'])
                    self.neuroticism_score_mae_list.append(calculate_obj.linear_regression_mae['neuroticism'])

                    # pearson score
                    self.openness_score_pearson_list.append(calculate_obj.linear_regression_pearson['openness'])
                    self.conscientiousness_score_pearson_list.append(calculate_obj.linear_regression_pearson['conscientiousness'])
                    self.extraversion_score_pearson_list.append(calculate_obj.linear_regression_pearson['extraversion'])
                    self.agreeableness_score_pearson_list.append(calculate_obj.linear_regression_pearson['agreeableness'])
                    self.neuroticism_score_pearson_list.append(calculate_obj.linear_regression_pearson['neuroticism'])

                    '''# CV score
                    self.openness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['openness'])
                    self.conscientiousness_cv_score_list.append(
                        calculate_obj.logistic_regression_accuracy_cv['conscientiousness'])
                    self.extraversion_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['extraversion'])
                    self.agreeableness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['agreeableness'])
                    self.neuroticism_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['neuroticism'])'''


                self.plot_traits_mae_versus_threshold_linear(cur_penalty, k_best)
                self.plot_traits_pearson_versus_threshold_linear(cur_penalty, k_best)

    # run experiments for a giving arguments
    def run_experiments(self, xgb_c, xgb_eta, xgb_max_depth):
        calculate_obj = CalculateScore(participant_file, item_aspects_file, purchase_history_file, valid_users_file,
                                       dir_analyze_name, self.threshold_purchase, self.bool_slice_gap_percentile,
                                       self.bool_normalize_features, self.C, self.cur_penalty,
                                       self.time_purchase_ratio_feature_flag, self.time_purchase_meta_feature_flag,
                                       self.vertical_ratio_feature_flag, self.purchase_percentile_feature_flag,
                                       self.user_meta_feature_flag, self.aspect_feature_flag, self.h_limit,
                                       self.l_limit, self.k_best, self.plot_directory, self.user_type,
                                       self.normalize_traits, self.classifier_type, self.split_bool, xgb_c, xgb_eta,
                                       xgb_max_depth, self.dir_logistic_results, self.cur_time, self.k_best_feature_flag)

        if not self.predefined_data_set_flag:

            # calculate_obj.init_debug_log()  # init log file
            calculate_obj.load_clean_csv_results()                  # load data set
            calculate_obj.clean_df()                                # clean df - e.g. remain valid users only
            calculate_obj.create_feature_list()                     # create x_feature

            calculate_obj.insert_gender_feature()                   # add gender feature
            calculate_obj.remove_except_cf()                        # remove not CF participants
            calculate_obj.extract_user_purchase_connection()        # insert purchase and vertical type to model

            if self.aspect_feature_flag:
                calculate_obj.extract_item_aspect()                 # add features of dominant item aspect

            calculate_obj.normalize_personality_trait()             # normalize trait to 0-1 scale (div by 5)

            # important!! after cut users by threshold
            calculate_obj.cal_participant_percentile_traits_values()  # calculate average traits and percentile value

            calculate_obj.insert_money_feature()                    # add feature contain money issue
            calculate_obj.insert_time_feature()                     # add time purchase feature
            calculate_obj.save_predefined_data_set()
        else:
            calculate_obj.create_feature_list()
            calculate_obj.load_predefined_data_set(self.predefined_data_set_path)

        if self.model_method == 'linear':
            calculate_obj.calculate_linear_regression()         # predict values 0-1, MAE and Pearson
        elif self.model_method == 'logistic':
            calculate_obj.calculate_logistic_regression()       # predict traits H or L

        return calculate_obj

    ############################# store in a CSV #############################

    def _store_data_df(self, model_results_array):
        """
        insert model result for a given configuration
        """
        """
        self.result_df = pd.DataFrame(columns=[
            'method', 'classifier', 'CV_bool', 'user_type', 'l_limit', 'h_limit',
            'threshold', 'k_features', 'xgb_gamma', 'xgb_eta', 'xgb_max_depth', 'trait', 'test_accuracy', 'auc',
            'accuracy_k_fold', 'auc_k_fold', 'train_accuracy', 'data_size', 'majority_ratio', 'features',
            'xgb_n_estimators', 'xgb_subsample', 'xgb_colsample_bytree'
        ])
        """
        for row in model_results_array:
            self.result_df = self.result_df.append({
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
            }, ignore_index=True)

            Logger.info('insert model number into result df: {}/{}, {}%'.format(
                self.result_df.shape[0],
                self.num_experiments,
                round(float(self.result_df.shape[0])/self.num_experiments, 2)*100
            ))

    def _save_result_df(self):
        """
        save all model results
        :return:
        """
        result_df_path = os.path.join(self.dir_logistic_results, 'compare_models')
        if not os.path.exists(result_df_path):
            os.makedirs(result_df_path)

        e = round(max(self.result_df.loc[self.result_df['trait'] == 'extraversion']['auc']), 2)
        o = round(max(self.result_df.loc[self.result_df['trait'] == 'openness']['auc']), 2)
        a = round(max(self.result_df.loc[self.result_df['trait'] == 'agreeableness']['auc']), 2)
        n = round(max(self.result_df.loc[self.result_df['trait'] == 'neuroticism']['auc']), 2)
        c = round(max(self.result_df.loc[self.result_df['trait'] == 'conscientiousness']['auc']), 2)
        best_acc = max(o, c, e, a, n)
        prefix_name = '{}_e={}_o={}_a={}_c={}_n={}_cnt={}_user={}_h={}_l={}'.format(
            best_acc, e, o, a, c, n, self.result_df.shape[0], self.user_type, self.h_limit, self.l_limit
        )

        result_df_path = os.path.join(result_df_path, '{}_{}.csv'.format(prefix_name, self.cur_time))
        self.result_df.to_csv(result_df_path, index=False)
        Logger.info('save result model: {}'.format(result_df_path))

    ############################# visualization functions #############################

    # plot traits accuracy versus - logistic
    def plot_traits_accuracy_versus_threshold(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt

        try:
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 2, 1)

            plt.figure(figsize=(10, 6))
            plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_cv_score_list, '.r:', label='openness CV')

            plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_cv_score_list, '.b:', label='conscientiousness CV')

            plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_cv_score_list, '.g:', label='extraversion CV')

            plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_cv_score_list, '.m:', label='agreeableness CV')

            plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_cv_score_list, '.c:', label='neuroticism CV')

            max_open = max(max(self.openness_score_list),
                           max(self.conscientiousness_score_list),
                           max(self.extraversion_score_list),
                           max(self.agreeableness_score_list),
                           max(self.neuroticism_score_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'traits test accuracy vs. amount purchase threshold \n'
            title += ' Max: ' + str(round(max_open, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(cur_penalty)\
                     + ' Gap:' + str(self.h_limit) + '-' + str(self.l_limit)

            plt.title(title)
            plt.ylabel('Test accuracy')
            plt.xlabel('Threshold purchase amount')
            # plt.ylim(0.4, 1)
            # plot_name = cur_directory + 'logistic_C=' + str(cur_C) + '_penalty=' \
            #             + str(cur_penalty) + '_max=' + str(round(max_open, 2)) + '_gap=' + str(
            #     bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '.png'
            plot_name = str(round(max_open, 2)) + '_Accuracy_k=' + str(k_best) + '_penalty=' + str(cur_penalty) + '_gap=' + str(
                self.h_limit) + '_' + str(self.l_limit) + '_max=' + str(round(max_open, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            Logger.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass

        return

    # plot traits accuracy versus - logistic
    def plot_traits_accuracy_versus_threshold_CV(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt

        try:
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 2, 1)

            plt.figure(figsize=(10, 6))
            # plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_cv_score_list, '.r:', label='openness CV')

            # plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_cv_score_list, '.b:', label='conscientiousness CV')

            # plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_cv_score_list, '.g:', label='extraversion CV')

            # plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_cv_score_list, '.m:', label='agreeableness CV')

            # plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_cv_score_list, '.c:', label='neuroticism CV')

            max_open = max(max(self.openness_cv_score_list),
                           max(self.conscientiousness_cv_score_list),
                           max(self.extraversion_cv_score_list),
                           max(self.agreeableness_cv_score_list),
                           max(self.neuroticism_cv_score_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'traits CV accuracy vs. amount purchase threshold \n'
            title += ' Max: ' + str(round(max_open, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(cur_penalty)\
                     + ' Gap:' + str(self.h_limit) + '-' + str(self.l_limit)

            plt.title(title)
            plt.ylabel('Test accuracy')
            plt.xlabel('Threshold purchase amount')
            # plt.ylim(0.4, 1)
            # plot_name = cur_directory + 'logistic_C=' + str(cur_C) + '_penalty=' \
            #             + str(cur_penalty) + '_max=' + str(round(max_open, 2)) + '_gap=' + str(
            #     bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '.png'
            plot_name = str(round(max_open, 2)) + '_CV_Accuracy_k=' + str(k_best) + '_penalty=' + str(cur_penalty) + '_gap=' + str(
                self.h_limit) + '_' + str(self.l_limit) + '_max=' + str(round(max_open, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            Logger.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass

        return

    # plot traits accuracy versus - logistic
    def plot_traits_roc_versus_threshold(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt

        try:
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 2, 1)

            plt.figure(figsize=(10, 6))
            plt.plot(self.threshold_list, self.openness_score_roc_list, '.r-', label='openness')
            # plt.plot(self.threshold_list, self.openness_cv_score_list, '.r:', label='openness CV')

            plt.plot(self.threshold_list, self.conscientiousness_score_roc_list, '.b-', label='conscientiousness')
            # plt.plot(self.threshold_list, self.conscientiousness_cv_score_list, '.b:', label='conscientiousness CV')

            plt.plot(self.threshold_list, self.extraversion_score_roc_list, '.g-', label='extraversion')
            # plt.plot(self.threshold_list, self.extraversion_cv_score_list, '.g:', label='extraversion CV')

            plt.plot(self.threshold_list, self.agreeableness_score_roc_list, '.m-', label='agreeableness')
            # plt.plot(self.threshold_list, self.agreeableness_cv_score_list, '.m:', label='agreeableness CV')

            plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            # plt.plot(self.threshold_list, self.neuroticism_cv_score_list, '.c:', label='neuroticism CV')

            max_open = max(max(self.openness_score_roc_list),
                           max(self.conscientiousness_score_roc_list),
                           max(self.extraversion_score_roc_list),
                           max(self.agreeableness_score_roc_list),
                           max(self.neuroticism_score_roc_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'traits test ROC score vs. amount purchase threshold \n'
            title += ' Max: ' + str(round(max_open, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(
                cur_penalty) \
                     + ' Gap:' + str(self.h_limit) + '-' + str(self.l_limit)

            plt.title(title)
            plt.ylabel('ROC Test score')
            plt.xlabel('Threshold purchase amount')
            # plt.ylim(0.4, 1)
            # plot_name = cur_directory + 'logistic_C=' + str(cur_C) + '_penalty=' \
            #             + str(cur_penalty) + '_max=' + str(round(max_open, 2)) + '_gap=' + str(
            #     bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '.png'
            plot_name = str(round(max_open, 2)) + '_ROC_k=' + str(k_best) + '_penalty=' + str(
                cur_penalty) + '_gap=' + str(
                self.h_limit) + '_' + str(self.l_limit) + '_max=' + str(round(max_open, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            Logger.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass

        return

    # plot results - linear regression TODO
    def plot_traits_mae_versus_threshold_linear(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt
        try:

            plt.figure(figsize=(10, 6))
            # plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_score_mae_list, '.r:', label='openness mae')

            # plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_score_mae_list, '.b:', label='conscientiousness mae')

            # plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_score_mae_list, '.g:', label='extraversion mae')

            # plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_score_mae_list, '.m:', label='agreeableness mae')

            # plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_score_mae_list, '.c:', label='neuroticism mae')

            min_mae = min(min(self.openness_score_mae_list),
                          min(self.conscientiousness_score_mae_list),
                          min(self.extraversion_score_mae_list),
                          min(self.agreeableness_score_mae_list),
                          min(self.neuroticism_score_mae_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'personalty traits MAE vs. amount purchase threshold \n'
            title += ' Min MAE: ' + str(round(min_mae, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(cur_penalty)


            plt.title(title)
            plt.ylabel('Test accuracy')
            plt.xlabel('Threshold purchase amount')
            plot_name = str(round(min_mae, 2)) + '_MAE_k=' + str(k_best) + '_penalty=' + str(cur_penalty) + '_min_mae=' + str(round(min_mae, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            Logger.info('min MAE: ' + str(round(min_mae, 3)))
            Logger.info('save plot: ' + str(plot_path))


        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass
        return

    # plot results - linear regression TODO
    def plot_traits_pearson_versus_threshold_linear(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt
        try:

            plt.figure(figsize=(10, 6))
            # plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_score_pearson_list, '.r:', label='openness pearson')

            # plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_score_pearson_list, '.b:', label='conscientiousness pearson')

            # plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_score_pearson_list, '.g:', label='extraversion pearson')

            # plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_score_pearson_list, '.m:', label='agreeableness pearson')

            # plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_score_pearson_list, '.c:', label='neuroticism pearson')

            max_p = max(max(self.openness_score_pearson_list),
                          max(self.conscientiousness_score_pearson_list),
                          max(self.extraversion_score_pearson_list),
                          max(self.agreeableness_score_pearson_list),
                          max(self.neuroticism_score_pearson_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'personalty traits Pearson vs. amount purchase threshold \n'
            title += ' max Pearson: ' + str(round(max_p, 3)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(
                cur_penalty)

            plt.title(title)
            plt.ylabel('Pearson correlation')
            plt.xlabel('Threshold purchase amount')
            plot_name = str(round(max_p, 3)) + '_Pearson_k=' + str(k_best) + '_penalty=' + str(
                cur_penalty) + '_max_pearson=' + str(round(max_p, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            Logger.info('max pearson: ' + str(round(max_p, 3)))
            Logger.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass
        return


def main():

    wrapper_obj = Wrapper()
    wrapper_obj.init_debug_log()        # init debug once - log file
    wrapper_obj.run_models()

if __name__ == '__main__':
    main()
