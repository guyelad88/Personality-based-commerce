from __future__ import print_function
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

from utils.logger import Logger
from config import bfi_config
from build_item_aspect_feature import BuildItemAspectScore

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import linear_model, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif


class CalculateScore:

    def __init__(self, participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
                 threshold_purchase, bool_slice_gap_percentile=True, bool_normalize_features=True, C=2,
                 cur_penalty='l1', time_purchase_ratio_feature_flag=True, time_purchase_meta_feature_flag=True,
                 vertical_ratio_feature_flag=True, purchase_percentile_feature_flag=True,
                 user_meta_feature_flag=True, aspect_feature_flag=True, h_limit=0.6, l_limit=0.4,
                 k_best=10, plot_directory='', user_type='all', normalize_traits=True, classifier_type='xgb',
                 split_bool=False, xgb_c=1, xgb_eta=0.1, xgb_max_depth=4, dir_logistic_results='', cur_time='',
                 k_best_feature_flag=True):

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

        self.models_results = list()        # store model result (later will insert into a result CSV)

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
        self.k_best = random.randint(8, 20)     # k_best  # number of k_best feature to select

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

        """
        self.xgb_n_estimators = bfi_config.predict_trait_configs['xgb_n_estimators']
        self.xgb_subsample = bfi_config.predict_trait_configs['xgb_subsample']
        self.xgb_colsample_bytree = bfi_config.predict_trait_configs['xgb_colsample_bytree']
        """

        self.pearson_relevant_feature = bfi_config.feature_data_set['pearson_relevant_feature']

        self.lr_y_feature = bfi_config.feature_data_set['lr_y_feature']
        self.lr_y_logistic_feature = bfi_config.feature_data_set['lr_y_logistic_feature']
        self.lr_y_linear_feature = bfi_config.feature_data_set['lr_y_linear_feature']
        self.trait_percentile = bfi_config.feature_data_set['trait_percentile']
        self.map_dict_percentile_group = bfi_config.feature_data_set['map_dict_percentile_group']

        self.time_purchase_ratio_feature_flag = time_purchase_ratio_feature_flag
        self.time_purchase_meta_feature_flag = time_purchase_meta_feature_flag
        self.vertical_ratio_feature_flag = vertical_ratio_feature_flag
        self.purchase_price_feature_flag = False                        # if true is overlap with purchase percentile
        self.purchase_percentile_feature_flag = purchase_percentile_feature_flag
        self.user_meta_feature_flag = user_meta_feature_flag
        self.aspect_feature_flag = aspect_feature_flag

        self.time_purchase_ratio_feature = bfi_config.feature_data_set['time_purchase_ratio_feature']
        self.time_purchase_meta_feature = bfi_config.feature_data_set['time_purchase_meta_feature']
        self.vertical_ratio_feature = bfi_config.feature_data_set['vertical_ratio_feature']
        self.purchase_price_feature = bfi_config.feature_data_set['purchase_price_feature']
        self.purchase_percentile_feature = bfi_config.feature_data_set['purchase_percentile_feature']
        self.user_meta_feature = bfi_config.feature_data_set['user_meta_feature']
        self.aspect_feature = bfi_config.feature_data_set['aspect_feature']

        measure_template = {
                'openness': 0.0,
                'conscientiousness': 0.0,
                'extraversion': 0.0,
                'agreeableness': 0.0,
                'neuroticism': 0.0
        }

        self.logistic_regression_accuracy = dict(measure_template)
        self.logistic_regression_roc = dict(measure_template)
        self.logistic_regression_accuracy_cv = dict(measure_template)
        self.linear_regression_mae = dict(measure_template)
        self.linear_regression_pearson = dict(measure_template)

    # load csv into df
    def load_clean_csv_results(self):

        self.participant_df = pd.read_csv(self.participant_file)
        self.item_aspects_df = pd.read_csv(self.item_aspects_file)
        self.purchase_history_df = pd.read_csv(self.purchase_history_file, error_bad_lines=False)
        self.valid_users_df = pd.read_csv(self.valid_users_file)

        return

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
        if self.purchase_price_feature_flag:
            self.lr_x_feature.extend(self.purchase_price_feature)
        if self.purchase_percentile_feature_flag:
            self.lr_x_feature.extend(self.purchase_percentile_feature)
        if self.user_meta_feature_flag:
            self.lr_x_feature.extend(self.user_meta_feature)
        if self.aspect_feature_flag:
            self.lr_x_feature.extend(self.aspect_feature)
        return

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
        return

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
        Logger.info('slice particpant below purchase threshold')
        Logger.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'purchase_amount_after_threshold.csv')

        # histogram of number of purchases
        plt.hist(histogram_purchase_list, bins=30)
        plt.title('Histogram of #purchase item per participants, #P ' + str(self.merge_df.shape[0]))
        plt.ylabel('Participant amount')
        plt.xlabel('#Purchases')
        plot_name = self.dir_analyze_name + 'histogram_purchases_per_user' + '_p_' + str(self.merge_df.shape[0]) + '_threshold_' + str(self.threshold_purchase) + '.png'
        plt.savefig(plot_name, bbox_inches='tight')
        # plt.show()
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

        Logger.info('')
        Logger.info('start to extract item aspect feature')

        item_aspect_obj = BuildItemAspectScore(self.item_aspects_df, self.participant_df, self.purchase_history_df,
                                               self.valid_users_df, self.merge_df, self.user_id_name_dict, self.aspect_feature)
        item_aspect_obj.add_aspect_features()
        self.merge_df = item_aspect_obj.merge_df
        Logger.info('number of features after add item aspect: {}'.format(self.merge_df.shape[1]))

    # TODO edit - add textual feature
    def textual_feature(self):
        # self.merge_df['title_length'] = np.nan
        return

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

    def calculate_pearson_correlation(self):

        corr_df = self.merge_df.corr(method='pearson')
        pearson_path = os.path.join(self.dir_logistic_results, 'corr', '{}.csv'.format(self.cur_time))
        corr_df.to_csv(pearson_path)
        Logger.info('')
        Logger.info('save pearson correlation df')
        Logger.info('Save file path: - {}'.format(pearson_path))

    def save_predefined_data_set(self):
        predefined_df_path = os.path.join(self.dir_logistic_results, 'pre_defined_df')
        if not os.path.exists(predefined_df_path):
            os.makedirs(predefined_df_path)

        predefined_df_path = os.path.join(predefined_df_path, 'shape={}_{}_time=_{}.csv'.format(
            self.merge_df.shape[0],
            self.merge_df.shape[1],
            self.cur_time
        ))
        self.merge_df.to_csv(predefined_df_path, index=False)
        Logger.info('save pre-defined data set shape: {}'.format(self.merge_df.shape))

    def load_predefined_data_set(self, predefined_path):
        self.merge_df = pd.read_csv(predefined_path)
        Logger.info('load pre-defined data set shape: {}'.format(self.merge_df.shape))

    # calculate logistic regression model
    def calculate_logistic_regression(self):

        # self.map_dict_percentile_group = dict(zip(self.lr_y_logistic_feature,  self.trait_percentile))

        self.models_results = list()     # contain all results for the 5 traits models

        # test score for each trait
        openness_score, conscientiousness_score, extraversion_score, agreeableness_score, neuroticism_score = \
            list(), list(), list(), list(), list()
        openness_score_cv, conscientiousness_score_cv, extraversion_score_cv, agreeableness_score_cv, neuroticism_score_cv = \
            list(), list(), list(), list(), list()
        openness_score_roc, conscientiousness_score_roc, extraversion_score_roc, agreeableness_score_roc, neuroticism_score_roc = \
            list(), list(), list(), list(), list()

        relevant_X_columns = copy.deepcopy(self.lr_x_feature)
        map_dict_feature_non_zero = dict()
        for trait in self.lr_y_logistic_feature:
            map_dict_feature_non_zero[trait] = dict(zip(list(relevant_X_columns), [0]*len(relevant_X_columns)))

        # add column H/L for each trait
        self.add_high_low_traits_column()

        # create model all target features

        # self.calculate_pearson_correlation()

        for idx, y_feature in enumerate(self.lr_y_logistic_feature):    # build model for each trait separately

            Logger.info('')
            Logger.info('build model for: {}'.format(str(y_feature)))

            X, y = self._prepare_raw_data_to_model(y_feature, relevant_X_columns)

            # whether to use cross validation or just train-test
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            majority_ratio = max(round(sum(y) / len(y), 2), 1 - round(sum(y) / len(y), 2))
            data_size = X.shape[0]

            assert self.k_best_feature_flag

            if self.k_best_feature_flag:
                X, X_train, X_test, k_feature = self._select_k_best_feature(X, y, X_train, y_train, X_test)

            if self.classifier_type == 'xgb':
                regr = XGBClassifier(
                    n_estimators=self.xgb_n_estimators,
                    max_depth=self.xgb_max_depth,
                    learning_rate=self.xgb_eta,
                    gamma=self.xgb_c,
                    subsample=self.xgb_subsample,
                    colsample_bytree=self.xgb_colsample_bytree
                )
                # subsample=0.8, colsample_bytree=1, gamma=1)

                kfold = StratifiedKFold(n_splits=4, random_state=7)
                acc_arr = cross_val_score(regr, X, y, cv=kfold)
                auc_arr = cross_val_score(regr, X, y, cv=kfold, scoring='roc_auc')

                acc_mean, acc_std = round(acc_arr.mean(), 2), round(acc_arr.std(), 2)
                auc_mean, auc_std = round(auc_arr.mean(), 2), round(auc_arr.std(), 2)

                # extract feature importance
                regr.fit(X, y)
                dict_importance = dict(zip(k_feature, regr.feature_importances_))
                dict_param = self._log_parameters_order(dict_importance)

                Logger.info("")
                Logger.info('Accuracy: {}'.format(round(acc_mean, 2)))
                Logger.info('AUC: {}'.format(str(round(auc_mean, 2))))
                Logger.info('Accuracy list: {}'.format(acc_arr))
                Logger.info('AUC list: {}'.format(auc_arr))
                Logger.info(regr)

                # dict_param = dict(zip(k_feature, regr.feature_importances_))

            elif self.classifier_type == 'lr':

                if self.split_bool:
                    Logger.info("implement logistic regression (without CV)")
                    regr = linear_model.LogisticRegression(
                        penalty=self.penalty,
                        C=self.C,
                        solver='liblinear'
                    )
                    regr.fit(X_train, y_train)
                    train_score = regr.score(X_train, y_train)
                    test_score = regr.score(X_test, y_test)
                    prob_test_score = regr.predict_proba(X_test)
                    y_1_prob = prob_test_score[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_1_prob)
                    auc_score = auc(fpr, tpr)

                    Logger.info("")
                    Logger.info('train_score: {}'.format(round(train_score, 3)))
                    Logger.info('test_score: {}'.format(round(test_score, 3)))
                    Logger.info('auc score: {}'.format(str(round(auc_score, 3))))

                    dict_param = dict(zip(k_feature, regr.coef_[0]))
                    dict_param['intercept'] = regr.intercept_

                else:
                    Logger.info("implement CV logistic regression")
                    regr = linear_model.LogisticRegressionCV(  # Cs=1,
                        penalty=self.penalty,
                        solver='liblinear',
                        cv=model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=None)
                    )
                    regr.fit(X_train, y_train)
                    c_index = np.where(regr.Cs_ == regr.C_[0])[0][0]
                    c = regr.C_[0], 3
                    train_score = sum(regr.scores_[1][:, c_index]) / 4  # num splits

                    Logger.info("")     # TODO log k fold CV score
                    Logger.info('CV score: ' + str(round(train_score, 3)))
                    Logger.info('C value: ' + str(round(regr.C_[0], 3)))

                    dict_param = dict(zip(k_feature, regr.coef_[0]))
                    dict_param['intercept'] = regr.intercept_[0]

                    # TODO: implement this
                    self.create_roc_cv_plt(X_train, y_train, regr.C_[0])

            else:
                raise ValueError('unknown classifier type - {}'.format(self.classifier_type))

            # dict_param = self._log_parameters_order(dict_param)
            # dict_param = {}
            # insert current model to store later in a CSV
            self._prepare_model_result(
                y_feature, acc_mean, auc_mean, '', dict_param, data_size, majority_ratio, acc_arr, auc_arr, X
            )

            if self.classifier_type == 'lr' and self.split_bool:
                if y_feature == 'openness_group':
                    openness_score.append(acc_mean)
                    openness_score_roc.append(auc_mean)
                if y_feature == 'conscientiousness_group':
                    conscientiousness_score.append(acc_mean)
                    conscientiousness_score_roc.append(auc_mean)
                if y_feature == 'extraversion_group':
                    extraversion_score.append(acc_mean)
                    extraversion_score_roc.append(auc_mean)
                if y_feature == 'agreeableness_group':
                    agreeableness_score.append(acc_mean)
                    agreeableness_score_roc.append(auc_mean)
                if y_feature == 'neuroticism_group':
                    neuroticism_score.append(acc_mean)
                    neuroticism_score_roc.append(auc_mean)

            if y_feature == 'openness_group':
                openness_score_cv.append(acc_mean)
            if y_feature == 'conscientiousness_group':
                conscientiousness_score_cv.append(acc_mean)
            if y_feature == 'extraversion_group':
                extraversion_score_cv.append(acc_mean)
            if y_feature == 'agreeableness_group':
                agreeableness_score_cv.append(acc_mean)
            if y_feature == 'neuroticism_group':
                neuroticism_score_cv.append(acc_mean)

        if self.split_bool:
            if len(openness_score):
                self.logistic_regression_accuracy['openness'] = (sum(openness_score) / len(openness_score))
            if len(conscientiousness_score):
                self.logistic_regression_accuracy['conscientiousness'] = (sum(conscientiousness_score) / len(conscientiousness_score))
            if len(extraversion_score):
                self.logistic_regression_accuracy['extraversion'] = (sum(extraversion_score) / len(extraversion_score))
            if len(agreeableness_score):
                self.logistic_regression_accuracy['agreeableness'] = (sum(agreeableness_score) / len(agreeableness_score))
            if len(neuroticism_score):
                self.logistic_regression_accuracy['neuroticism'] = (sum(neuroticism_score) / len(neuroticism_score))

            if len(openness_score):
                self.logistic_regression_roc['openness'] = (sum(openness_score_roc) / len(openness_score_roc))
            if len(conscientiousness_score):
                self.logistic_regression_roc['conscientiousness'] = (sum(conscientiousness_score_roc) / len(conscientiousness_score_roc))
            if len(extraversion_score):
                self.logistic_regression_roc['extraversion'] = (sum(extraversion_score_roc) / len(extraversion_score_roc))
            if len(agreeableness_score):
                self.logistic_regression_roc['agreeableness'] = (sum(agreeableness_score_roc) / len(agreeableness_score_roc))
            if len(neuroticism_score):
                self.logistic_regression_roc['neuroticism'] = (sum(neuroticism_score_roc) / len(neuroticism_score_roc))

        if len(openness_score_cv):
            self.logistic_regression_accuracy_cv['openness'] = (sum(openness_score_cv) / len(openness_score_cv))
        if len(conscientiousness_score_cv):
            self.logistic_regression_accuracy_cv['conscientiousness'] = (sum(conscientiousness_score_cv) / len(conscientiousness_score_cv))
        if len(extraversion_score_cv):
            self.logistic_regression_accuracy_cv['extraversion'] = (sum(extraversion_score_cv) / len(extraversion_score_cv))
        if len(agreeableness_score_cv):
            self.logistic_regression_accuracy_cv['agreeableness'] = (sum(agreeableness_score_cv) / len(agreeableness_score_cv))
        if len(neuroticism_score_cv):
            self.logistic_regression_accuracy_cv['neuroticism'] = (sum(neuroticism_score_cv) / len(neuroticism_score_cv))

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
            'k_features': X.shape[1] if self.k_best_feature_flag else self.k_best,
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

        if self.bool_slice_gap_percentile:
            cur_f = self.map_dict_percentile_group[y_feature]
            self.merge_df.to_csv('{}logistic_regression_merge_df.csv'.format(self.dir_analyze_name))

            h_df = self.merge_df.loc[self.merge_df[cur_f] >= self.h_limit]
            l_df = self.merge_df.loc[self.merge_df[cur_f] <= self.l_limit]

            Logger.info('H group amount: ' + str(h_df.shape[0]))
            Logger.info('L group amount: ' + str(l_df.shape[0]))

            frames = [l_df, h_df]
            self.raw_df = pd.concat(frames)
        else:
            self.raw_df = self.merge_df

        self.raw_df = self.raw_df.fillna(0)
        self.raw_df.to_csv('{}lr_final_data.csv'.format(self.dir_analyze_name))

        Logger.info('')
        Logger.info('Save file: self.raw_df - {} lr_final_data.csv'.format(str(self.dir_analyze_name)))

        relevant_X_columns = copy.deepcopy(self.lr_x_feature)
        if y_feature in relevant_X_columns:
            relevant_X_columns.remove(y_feature)

        self.raw_df = self.raw_df[relevant_X_columns + [y_feature]]

        if self.bool_normalize_features:
            self.raw_df = self.preprocessing_min_max(self.raw_df)

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

    def _select_k_best_feature(self, X, y, X_train, y_train, X_test):
        """
        select K-best feature and transform feature set to k-feature.
        """
        k_model = SelectKBest(f_classif, k=self.k_best).fit(X_train, y_train)

        idxs_selected = k_model.get_support(indices=True)
        k_feature = list()

        for idx, cur_feature in enumerate(X):
            if idx in idxs_selected:
                k_feature.append(cur_feature)

        X_train = k_model.transform(X_train)
        X = k_model.transform(X)

        if self.split_bool:
            X_test = k_model.transform(X_test)

            assert X_train.shape[1] == X_test.shape[1]

        Logger.info('Total sample size: {}'.format(self.raw_df.shape[0]))
        Logger.info('Number of features before selecting: {}'.format(self.raw_df.shape[1]))
        Logger.info('Number of k best features: {}'.format(X_train.shape[1]))
        Logger.info('K feature selected: {}'.format(', '.join(k_feature)))

        return X, X_train, X_test, k_feature

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

    def _create_auc_plot(self, fpr, tpr, auc_score, X_test, y_test, y_feature):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(str(y_feature) + ': test amount ' + str(X_test.shape[0]) + ', test prop ' + str(
            round(sum(y_test) / len(y_test), 2)))
        plt.legend(loc="lower right")

        plot_name = str(round(auc_score, 2)) + '_ROC_k=' + str(self.k_best) + '_penalty=' + str(
            self.penalty) + '_gap=' + str(
            self.h_limit) + '_' + str(self.l_limit) + '_test_amount=' + \
                    str(X_test.shape[0]) + '_threshold=' + str(self.threshold_purchase) + '_trait=' + str(
            y_feature) + '_max=' + str(round(auc_score, 2)) + '.png'

        if not os.path.exists(self.plot_directory + '/roc/'):
            os.makedirs(self.plot_directory + '/roc/')

        plot_path = self.plot_directory + '/roc/' + plot_name
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    # calculate linear regression regression model
    def calculate_linear_regression(self):

        test_score = list()
        train_score_check = list()
        test_score_check = list()

        # test score for each trait

        openness_score_mae = list()
        conscientiousness_score_mae = list()
        extraversion_score_mae = list()
        agreeableness_score_mae = list()
        neuroticism_score_mae = list()

        openness_score_pearson = list()
        conscientiousness_score_pearson = list()
        extraversion_score_pearson = list()
        agreeableness_score_pearson = list()
        neuroticism_score_pearson = list()

        # train cv score for each trait
        openness_score_cv = list()
        conscientiousness_score_cv = list()
        extraversion_score_cv = list()
        agreeableness_score_cv = list()
        neuroticism_score_cv = list()

        # ROC test score for each trait
        openness_score_roc = list()
        conscientiousness_score_roc = list()
        extraversion_score_roc = list()
        agreeableness_score_roc = list()
        neuroticism_score_roc = list()

        import copy
        relevant_X_columns = copy.deepcopy(self.lr_x_feature)

        from sklearn import linear_model
        from sklearn import model_selection
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        from sklearn.feature_selection import SelectKBest
        from scipy.stats import pearsonr


        for i in range(0, 1):  # iterate N iterations
            for idx, y_feature in enumerate(self.lr_y_feature):    # iterate each trait
                Logger.info('')
                Logger.info('build linear regression model for: ' + str(y_feature))

                self.raw_df = self.merge_df
                self.raw_df.to_csv(self.dir_analyze_name + 'linear_regression_final_data.csv')
                Logger.info('')
                Logger.info('save file: ')

                import copy
                relevant_X_columns = copy.deepcopy(self.lr_x_feature)
                if y_feature in relevant_X_columns:
                    relevant_X_columns.remove(y_feature)

                self.raw_df = self.raw_df[relevant_X_columns + [y_feature]]

                # create corr df -> features and personality traits
                if False:
                    self.calculate_pearson_correlation(relevant_X_columns, y_feature, self.raw_df)

                # TODO normalize without target column
                # if self.bool_normalize_features:
                #    self.raw_df = self.preprocessing_min_max(self.raw_df)

                X = self.raw_df[relevant_X_columns]
                y = self.raw_df[y_feature]

                self.split_bool = True
                if self.split_bool:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=self.test_fraction
                    )
                    Logger.info('train: class 0 ratio: ' + str(sum(y_train) / len(y_train)))
                    Logger.info('test: class 0 ratio: ' + str(sum(y_test) / len(y_test)))
                else:
                    X_train = X
                    y_train = y

                Logger.info('all: class 0 ratio:  ' + str(sum(y)/len(y)))

                '''train_df, test_df = train_test_split(
                    self.raw_df,
                    test_size=self.test_fraction
                )

                X_train = train_df[relevant_X_columns]
                y_train = train_df[y_feature]
                X_test = test_df[relevant_X_columns]
                y_test = test_df[y_feature]
                '''

                X_train_old = copy.deepcopy(X_train)
                y_train_old = copy.deepcopy(y_train)

                if True:
                    from sklearn.feature_selection import f_classif
                    k_model = SelectKBest(f_classif, k=self.k_best).fit(X_train, y_train)

                    idxs_selected = k_model.get_support(indices=True)
                    k_feature = list()
                    # features_dataframe_new = X_train[idxs_selected]
                    for idx, cur_feature in enumerate(X):
                        if idx in idxs_selected:
                            k_feature.append(cur_feature)

                    X_train = k_model.transform(X_train)

                    if self.split_bool:
                        X_test = k_model.transform(X_test)

                        assert X_train.shape[1] == X_test.shape[1]

                # X_train.to_csv(self.dir_analyze_name + 'logistic_regression_df.csv')
                # y_train.to_csv(self.dir_analyze_name + 'logistic_regression_y_df.csv')
                Logger.info('')
                Logger.info('Total sample size: ' + str(self.raw_df.shape[0]))
                Logger.info('Number of features before selecting: ' + str(self.raw_df.shape[1]))
                Logger.info('Number of k best features: ' + str(X_train.shape[1]))

                from sklearn import datasets, linear_model
                from sklearn.model_selection import cross_validate
                from sklearn.metrics.scorer import make_scorer
                from sklearn.metrics import confusion_matrix
                from sklearn.svm import LinearSVC
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_absolute_error
                from sklearn.metrics import r2_score
                from sklearn.svm import SVC
                from sklearn.linear_model import RidgeCV, Ridge


                '''regr = cross_validate(
                    LinearRegression(),
                    X_train,
                    y_train,
                    cv=6,
                    scoring='r2', # 'mean_absolute_error',
                    return_train_score=True,
                    verbose=2
                )'''

                # regr = RidgeCV(normalize=True)
                regr = Ridge(normalize=True)
                '''regr = LinearRegression(
                     normalize=True
                )'''

                regr.fit(X_train, y_train)

                # train_score = regr.score(X_train, y_train)
                if self.split_bool:

                    y_pred_test = regr.predict(X_test)
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    pearson_c_test, p_value_test = pearsonr(y_test, y_pred_test)

                    y_pred_train = regr.predict(X_train)
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    pearson_c_train, p_value_train = pearsonr(y_train, y_pred_train)
                    train_score = mae_train

                    mae_threshold = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))

                    Logger.info('')
                    Logger.info('MAE train: ' + str(mae_train))
                    Logger.info('MAE test: ' + str(mae_test))
                    Logger.info('MAE threshold: ' + str(mae_threshold))

                    Logger.info('Pearson train: ' + str(round(pearson_c_train, 2)) + ', p val: ' + str(round(p_value_train, 3)))
                    Logger.info('Pearson test: ' + str(round(pearson_c_test, 2)) + ', p val: ' + str(round(p_value_test, 3)))
                    Logger.info('')

                    min_s = min(min(y_pred_test), min(y_pred_train), min(y_test), min(y_train))
                    max_s = max(max(y_pred_test), max(y_pred_train), max(y_test), max(y_train))

                    plt.figure(figsize=(11, 6))
                    plt.subplot(1, 2, 1)
                    plt.scatter(y_test, y_pred_test) # , s=area, c=colors, alpha=0.5)
                    # plt.plot([0, 1], [0, 1], color='navy')
                    plt.plot([min_s, max_s], [min_s, max_s], color='navy')
                    plt.xlim([min_s, max_s])
                    plt.ylim([min_s, max_s])
                    # plt.xlim([0.4, 1.0])
                    # plt.ylim([0.4, 1.05])
                    plt.xlabel('Y test')
                    plt.ylabel('Y predicted')
                    plt.title(str(y_feature) + ' ' + 'MAE test: ' + str(round(mae_test, 3)) + '\n Pe c: ' + str(round(pearson_c_test, 2)) + ' P_val: ' + str(round(p_value_test, 3)))
                    plt.legend(loc="lower right")

                    plt.subplot(1, 2, 2)
                    plt.scatter(y_train, y_pred_train)  # , s=area, c=colors, alpha=0.5)
                    plt.plot([min_s, max_s], [min_s, max_s], color='navy')
                    plt.xlim([min_s, max_s])
                    plt.ylim([min_s, max_s])
                    plt.xlabel('Y test')
                    plt.ylabel('Y predicted')
                    plt.title(str(y_feature) + ' ' + 'MAE train: ' + str(round(mae_train, 3)) + '\n Pe c: ' + str(round(pearson_c_train, 2)) + ' P_val: ' + str(round(p_value_train, 3)))
                    plt.legend(loc="lower right")
                    # plt.show()

                    plot_name = str(round(pearson_c_test, 3)) + '_Pearson_MAE=' + str(round(mae_test, 3)) + '_k=' + str(self.k_best) + '_penalty=' + str(
                        self.penalty) + '_test_amount=' + \
                                str(X_test.shape[0]) + '_threshold=' + str(self.threshold_purchase) + '_trait=' + str(
                        y_feature) + '.png'

                    import os
                    if not os.path.exists(self.plot_directory + '/MAE_Pearson/'):
                        os.makedirs(self.plot_directory + '/MAE_Pearson/')

                    plot_path = self.plot_directory + '/MAE_Pearson/' + plot_name
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()

                dict_param = dict(zip(k_feature, regr.coef_))
                dict_param['intercept'] = regr.intercept_
                # print(dictionary)
                d_view = [(v, k) for k, v in dict_param.iteritems()]
                d_view.sort(reverse=True)  #

                Logger.info("")
                Logger.info("Model Parameters:")
                # sorted(((v, k) for k, v in dict_param.iteritems()), reverse=True)
                for v, k in d_view:
                    if v != 0:
                        Logger.info("%s: %f" % (k, v))
                Logger.info("")

                if self.split_bool:
                    if y_feature == 'openness_trait':
                        openness_score_mae.append(mae_test)
                        openness_score_pearson.append(pearson_c_test)
                    if y_feature == 'conscientiousness_trait':
                        conscientiousness_score_mae.append(mae_test)
                        conscientiousness_score_pearson.append(pearson_c_test)
                    if y_feature == 'extraversion_trait':
                        extraversion_score_mae.append(mae_test)
                        extraversion_score_pearson.append(pearson_c_test)
                    if y_feature == 'agreeableness_trait':
                        agreeableness_score_mae.append(mae_test)
                        agreeableness_score_pearson.append(pearson_c_test)
                    if y_feature == 'neuroticism_trait':
                        neuroticism_score_mae.append(mae_test)
                        neuroticism_score_pearson.append(pearson_c_test)
                '''if y_feature == 'openness_group':
                    openness_score_cv.append()
                if y_feature == 'conscientiousness_group':
                    conscientiousness_score_cv.append(train_score)
                if y_feature == 'extraversion_group':
                    extraversion_score_cv.append(train_score)
                if y_feature == 'agreeableness_group':
                    agreeableness_score_cv.append(train_score)
                if y_feature == 'neuroticism_group':
                    neuroticism_score_cv.append(train_score)'''
        if self.split_bool:
            if len(openness_score_mae):
                self.linear_regression_mae['openness'] = (sum(openness_score_mae) / len(openness_score_mae))
            if len(conscientiousness_score_mae):
                self.linear_regression_mae['conscientiousness'] = (
                sum(conscientiousness_score_mae) / len(conscientiousness_score_mae))
            if len(extraversion_score_mae):
                self.linear_regression_mae['extraversion'] = (sum(extraversion_score_mae) / len(extraversion_score_mae))
            if len(agreeableness_score_mae):
                self.linear_regression_mae['agreeableness'] = (sum(agreeableness_score_mae) / len(agreeableness_score_mae))
            if len(neuroticism_score_mae):
                self.linear_regression_mae['neuroticism'] = (sum(neuroticism_score_mae) / len(neuroticism_score_mae))

            if len(openness_score_pearson):
                self.linear_regression_pearson['openness'] = (sum(openness_score_pearson) / len(openness_score_pearson))
            if len(conscientiousness_score_pearson):
                self.linear_regression_pearson['conscientiousness'] = (
                sum(conscientiousness_score_pearson) / len(conscientiousness_score_pearson))
            if len(extraversion_score_pearson):
                self.linear_regression_pearson['extraversion'] = (sum(extraversion_score_pearson) / len(extraversion_score_pearson))
            if len(agreeableness_score_pearson):
                self.linear_regression_pearson['agreeableness'] = (sum(agreeableness_score_pearson) / len(agreeableness_score_pearson))
            if len(neuroticism_score_pearson):
                self.linear_regression_pearson['neuroticism'] = (sum(neuroticism_score_pearson) / len(neuroticism_score_pearson))

        '''if len(openness_score_cv):
            self.logistic_regression_accuracy_cv['openness'] = (sum(openness_score_cv) / len(openness_score_cv))
        if len(conscientiousness_score_cv):
            self.logistic_regression_accuracy_cv['conscientiousness'] = (sum(conscientiousness_score_cv) / len(conscientiousness_score_cv))
        if len(extraversion_score_cv):
            self.logistic_regression_accuracy_cv['extraversion'] = (sum(extraversion_score_cv) / len(extraversion_score_cv))
        if len(agreeableness_score_cv):
            self.logistic_regression_accuracy_cv['agreeableness'] = (sum(agreeableness_score_cv) / len(agreeableness_score_cv))
        if len(neuroticism_score_cv):
            self.logistic_regression_accuracy_cv['neuroticism'] = (sum(neuroticism_score_cv) / len(neuroticism_score_cv))'''

        # from collections import Counter

        # Logger.info('Counter C: ' + str(Counter(c_check)))

        '''print(true_list)
        print(len(true_list))
        print(false_list)
        print(len(false_list))'''

        # print('total ratio: ' + str(float(len(true_list))/float(len(true_list)+len(false_list))))
        return

    def create_roc_cv_plt(self, X_train, y_train, C):
        return
        import numpy as np
        from scipy import interp
        import matplotlib.pyplot as plt
        from itertools import cycle

        from sklearn import svm, datasets
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import StratifiedKFold
        from sklearn import linear_model
        X = X_train
        y = y_train

        # #############################################################################
        # Data IO and generation

        # Import some data to play with
        '''iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X, y = X[y != 2], y[y != 2]
        n_samples, n_features = X.shape

        # Add noisy features
        random_state = np.random.RandomState(0)
        X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]'''

        # #############################################################################
        # Classification and ROC analysis

        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=4, shuffle=True)
        classifier = linear_model.LogisticRegression(  # Cs=1,
            C = C,
            penalty=self.penalty,
            solver='liblinear')
        regr = linear_model.LogisticRegression(penalty=self.penalty, C=self.C)
        '''cv=model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=None))
        classifier = svm.SVC(kernel='linear', probability=True,
                             random_state=random_state)'''

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(X, y):
            import copy
            X_train = X[train]
            X_train = copy.deepcopy(X_train)

            y_train = y[train]
            y_train = copy.deepcopy(y_train)
            # y_train.reshape(y_train.shape[0], y_train.shape[1])
            X_test = X[test]
            X_test = copy.deepcopy(X_test)

            y_test = y[test]
            y_test = copy.deepcopy(y_test)
            # y_test.reshape(y_test.shape[0], y_test.shape[1])
            probas_ = regr.fit(X_train, y_train).predict_proba(X_test)
            # probas_ = regr.fit(X, y).predict_proba(X)

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            # fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        return

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
            # Logger.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self._cal_participant_traits_percentile_values(idx, row_participant)

        # self.merge_df = self.participant_df.copy()

        return

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


def main(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
         threshold_purchase, bool_slice_gap_percentile, bool_normalize_features, c_value, regularization):

    calculate_obj = CalculateScore(participant_file, item_aspects_file, purchase_history_file, valid_users_file,
                                   dir_analyze_name, threshold_purchase, bool_slice_gap_percentile,
                                   bool_normalize_features, c_value, regularization)    # create object and variables

    calculate_obj.load_clean_csv_results()              # load data set
    calculate_obj.clean_df()                            # clean df - e.g. remain valid users only
    calculate_obj.create_feature_list()                 # create x_feature (structure only)

    # calculate personality trait per user + percentile per trait
    calculate_obj.change_reverse_value()                # change specific column into reverse mode
    calculate_obj.cal_participant_traits_values()       # calculate average traits and percentile value
    calculate_obj.insert_gender_feature()               # add gender feature

    calculate_obj.extract_user_purchase_connection()    # insert purchase and vertical type to model
    calculate_obj.insert_money_feature()                # add feature contain money issue
    calculate_obj.insert_time_feature()                 # add time purchase feature

    # calculate_obj.textual_feature()                       # TODO even basic feature
    calculate_obj.extract_item_aspect()                     # TODO add
    # calculate_obj.cal_all_participant_percentile_value()  # TODO need to add

    calculate_obj.calculate_logistic_regression()       # predict traits H or L
    # calculate_obj.calculate_linear_regression()         # predict traits using other feature
    # calculate_obj.calculate_pearson_correlation()       # calculate pearson correlation


if __name__ == '__main__':

    raise SystemExit('not in use - please run using Wrapper_build_feature_dataset')

