from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from p_values_for_logreg import LogisticReg

class CalculateScore:

    def __init__(self, participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
                 threshold_purchase, bool_slice_gap_percentile=True, bool_normalize_features=True, cur_C=2,
                 cur_penalty='l1', time_purchase_ratio_feature_flag=True, time_purchase_meta_feature_flag=True,
                                        vertical_ratio_feature_flag=True, purchase_percentile_feature_flag=True,
                                        user_meta_feature_flag=True, h_limit=0.6, l_limit=0.4):

        # file arguments
        self.participant_file = participant_file
        self.item_aspects_file = item_aspects_file
        self.purchase_history_file = purchase_history_file
        self.valid_users_file = valid_users_file
        self.dir_analyze_name = dir_analyze_name

        # define data frame needed for analyzing data
        self.participant_df = pd.DataFrame()
        self.item_aspects_df = pd.DataFrame()
        self.purchase_history_df = pd.DataFrame()
        self.valid_users_df = pd.DataFrame()

        self.merge_df = pd.DataFrame()                  # merge df - final for analyze and correlation
        self.raw_df = pd.DataFrame()                    # raw data df using in prediction function

        self.avg_openness = 0
        self.avg_conscientiousness = 0
        self.avg_extraversion = 0
        self.avg_agreeableness = 0
        self.avg_neuroticism = 0

        self.ratio_hundred_openness = 0
        self.ratio_hundred_conscientiousness = 0
        self.ratio_hundred_extraversion = 0
        self.ratio_hundred_agreeableness = 0
        self.ratio_hundred_neuroticism = 0

        self.question_openness = [5, 10, 15, 20, 25, 30, 35, 40, 41, 44]
        self.question_conscientiousness = [3, 8, 13, 18, 23, 28, 33, 43]
        self.question_extraversion = [1, 6, 11, 16, 21, 26, 31, 36]
        self.question_agreeableness = [2, 7, 12, 17, 22, 27, 32, 37, 42]
        self.question_neuroticism = [4, 9, 14, 19, 24, 29, 34, 39]

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

        self.valid_user_list = list()   # valid users

        self.item_buyer_dict = dict()       # item-buyer dict 1:N
        self.user_name_id_dict = dict()     # missing data because it 1:N
        self.user_id_name_dict = dict()     #

        # system hyper_parameter
        self.threshold_purchase = threshold_purchase    # throw below this number
        self.C = cur_C
        self.penalty = cur_penalty
        self.bool_slice_gap_percentile = bool_slice_gap_percentile
        self.bool_normalize_features = bool_normalize_features
        self.threshold_pearson = 0.2
        self.test_fraction = 0.2
        self.h_limit = h_limit


        self.l_limit = l_limit
        self.pearson_relevant_feature = ['Age', 'openness_percentile',
                   'conscientiousness_percentile', 'extraversion_percentile', 'agreeableness_percentile',
                   'neuroticism_percentile', 'number_purchase', 'Electronics_ratio', 'Fashion_ratio',
                   'Home & Garden_ratio', 'Collectibles_ratio', 'Lifestyle_ratio', 'Parts & Accessories_ratio',
                   'Business & Industrial_ratio', 'Media_ratio']

        self.lr_x_feature = list()

        self.lr_y_feature = ['agreeableness_trait', 'extraversion_trait', 'neuroticism_trait', 'conscientiousness_trait', 'openness_trait'
                             ]

        self.lr_y_logistic_feature = ['openness_group', 'conscientiousness_group', 'extraversion_group','agreeableness_group', 'neuroticism_group'] #['neuroticism_group']#
        self.lr_y_logistic_feature = ['conscientiousness_group']  # ['neuroticism_group']#

        self.trait_percentile = ['openness_percentile', 'conscientiousness_percentile', 'extraversion_percentile',
                                      'agreeableness_percentile', 'neuroticism_percentile']

        self.map_dict_percentile_group = {
            'extraversion_group': 'extraversion_percentile',
            'openness_group': 'openness_percentile',
            'conscientiousness_group': 'conscientiousness_percentile',
            'agreeableness_group': 'agreeableness_percentile',
            'neuroticism_group': 'neuroticism_percentile'

        }

        self.time_purchase_ratio_feature_flag = time_purchase_ratio_feature_flag
        self.time_purchase_meta_feature_flag = time_purchase_meta_feature_flag
        self.vertical_ratio_feature_flag = vertical_ratio_feature_flag
        self.purchase_price_feature_flag = False
        self.purchase_percentile_feature_flag = purchase_percentile_feature_flag
        self.user_meta_feature_flag = user_meta_feature_flag

        self.time_purchase_ratio_feature = ['day_ratio', 'evening_ratio', 'night_ratio']#, 'weekend_ratio']
        self.time_purchase_meta_feature = ['first_purchase', 'last_purchase', 'tempo_purchase']

        self.vertical_ratio_feature = ['Electronics_ratio', 'Fashion_ratio', 'Home & Garden_ratio', 'Collectibles_ratio',
                               'Lifestyle_ratio', 'Parts & Accessories_ratio', 'Business & Industrial_ratio',
                               'Media_ratio']

        self.purchase_price_feature = ['median_purchase_price', 'q1_purchase_price', 'q3_purchase_price',
                                 'min_purchase_price', 'max_purchase_price']

        self.purchase_percentile_feature = ['median_purchase_price_percentile', 'q1_purchase_price_percentile',
                                    'q3_purchase_price_percentile', 'min_purchase_price_percentile',
                                    'max_purchase_price_percentile']

        self.user_meta_feature = ['Age', 'gender', 'number_purchase']

        self.logistic_regression_accuracy = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

    # build log object
    def init_debug_log(self):
        import logging
        logging.basicConfig(filename='/Users/sguyelad/PycharmProjects/research/survey_pilot/log/analyze_results.log',
                            filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info("")
        logging.info("")
        logging.info("start log program")

    # load csv into df
    def load_clean_csv_results(self):

        self.participant_df = pd.read_csv(self.participant_file)
        self.item_aspects_df = pd.read_csv(self.item_aspects_file)
        self.purchase_history_df = pd.read_csv(self.purchase_history_file)
        self.valid_users_df = pd.read_csv(self.valid_users_file)

        return

    def clean_df(self):
        # use only valid user id
        tmp_valid_user_list = list(self.valid_users_df['USER_SLCTD_ID'])
        self.valid_user_list = [x for x in tmp_valid_user_list if str(x) != 'nan']

        # extract only valid user name
        for (idx, row_participant) in self.participant_df.iterrows():
            # func = lambda s: s[:1].lower() + s[1:] if s else ''
            lower_first_name = row_participant['eBay site user name'].lower()
            self.participant_df.set_value(idx, 'eBay site user name', lower_first_name)

        self.participant_df = self.participant_df[self.participant_df['eBay site user name'].isin(self.valid_user_list)]

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
        return

    # extract user history purchase - amount
    # a. connect to number of purchases
    # b. connect to purchase after vertical
    def extract_user_purchase_connection(self):

        user_id_name_dict, histogram_purchase_list = self.insert_purchase_amount_data()
        self.slice_participant_using_threshold(histogram_purchase_list)
        self.insert_purchase_vertical_data(user_id_name_dict)

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df.csv')

        return

    # TODO edit - add textual feature
    def textual_feature(self):
        # self.merge_df['title_length'] = np.nan
        return

    def insert_gender_feature(self):
        self.merge_df['gender'] = \
            np.where(self.merge_df['Gender'] == 'Male', 1, 0)

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_gender.csv')
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

        # iterate over each user
        for buyer_id, group in price_group:
            # print(str(buyer_id) + ': ' + str(group.shape[0]))
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
            purchase_time_list = group['TRX_Timestamp'].tolist()
            cnt_non_weekend = 0
            for cur_per in purchase_time_list:

                time_object = time.strptime(cur_per, '%d/%m/%Y %H:%M')
                # cal first/last purchase and tempo
                if time_object.tm_year < user_count_type['first_year'] or (time_object.tm_year == user_count_type['first_year'] and time_object.tm_yday < user_count_type['first_day']):
                    user_count_type['first_year'] = time_object.tm_year
                    user_count_type['first_day'] = time_object.tm_yday

                if time_object.tm_year > user_count_type['last_year'] or (time_object.tm_year == user_count_type['last_year'] and time_object.tm_yday > user_count_type['last_day']):
                    user_count_type['last_year'] = time_object.tm_year
                    user_count_type['last_day'] = time_object.tm_yday

                # insert time in day/week
                correct_hour = (time_object.tm_hour + 9) % 24       # DB + 9 hour to fit israel time-zone
                if time_object.tm_wday in [4, 5]:
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

            cur_user_name = self.user_id_name_dict[buyer_id]
            if cur_user_name not in self.merge_df['eBay site user name'].tolist():
                continue
            cur_idx = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]

            self.merge_df.set_value(cur_idx, 'day_ratio', user_count_type['day_ratio'])
            self.merge_df.set_value(cur_idx, 'evening_ratio', user_count_type['evening_ratio'])
            self.merge_df.set_value(cur_idx, 'night_ratio', user_count_type['night_ratio'])
            self.merge_df.set_value(cur_idx, 'weekend_ratio', user_count_type['weekend_ratio'])
            self.merge_df.set_value(cur_idx, 'first_purchase', user_count_type['first_purchase'])
            self.merge_df.set_value(cur_idx, 'last_purchase', user_count_type['last_purchase'])
            self.merge_df.set_value(cur_idx, 'tempo_purchase', user_count_type['tempo_purchase'])

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_time_purchase.csv')

        '''self.merge_df.plot.scatter(x='gender', y='q1_purchase_price', figsize=(9, 7))
        plt.title('Age ' + ' vs. ' + 'median_purchase_price' + ', pearson corr = ')
        plt.show()
        plt.close()'''

        '''
        self.merge_df.plot.scatter(x='night_ratio', y='neuroticism_trait', figsize=(9, 7))
        plt.title('night_ratio' + ' vs. ' + 'neuroticism_trait' + ', pearson corr = ')
        plt.show()
        plt.close()
        '''

        return

    # add price features - value and percentile
    def insert_money_feature(self):

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

        self.add_price_feature()                # insert value feature
        self.add_percentile_price_feature()     # insert percentile feature

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_cost_value_percentile.csv')
        return

    def add_price_feature(self):
        # Price_USD
        price_group = self.purchase_history_df.groupby(['buyer_id'])
        for buyer_id, group in price_group:
            # print(str(buyer_id) + ': ' + str(group.shape[0]))

            cur_user_name = self.user_id_name_dict[buyer_id]
            if cur_user_name not in self.merge_df['eBay site user name'].tolist():
                continue

            user_percentile_price = group['Price_USD'].quantile([0, .25, .5, 0.75, 1])
            cur_user_name = self.user_id_name_dict[buyer_id]
            cur_idx = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]

            self.merge_df.set_value(cur_idx, 'min_purchase_price', user_percentile_price[0])
            self.min_cost_list.append(user_percentile_price[0])
            self.merge_df.set_value(cur_idx, 'q1_purchase_price', user_percentile_price[0.25])
            self.q1_cost_list.append(user_percentile_price[0.25])
            self.merge_df.set_value(cur_idx, 'median_purchase_price', user_percentile_price[0.5])
            self.median_cost_list.append(user_percentile_price[0.5])
            self.merge_df.set_value(cur_idx, 'q3_purchase_price', user_percentile_price[0.75])
            self.q3_cost_list.append(user_percentile_price[0.75])
            self.merge_df.set_value(cur_idx, 'max_purchase_price', user_percentile_price[1])
            self.max_cost_list.append(user_percentile_price[1])

        return

    def add_percentile_price_feature(self):

        for (idx, row_participant) in self.merge_df.iterrows():

            min_per = float(sum(i < row_participant['min_purchase_price'] for i in self.min_cost_list)) / float(
                len(self.min_cost_list) - 1)
            self.merge_df.set_value(idx, 'min_purchase_price_percentile', min_per)

            q1_per = float(sum(i < row_participant['q1_purchase_price'] for i in self.q1_cost_list)) / float(
                len(self.q1_cost_list) - 1)
            self.merge_df.set_value(idx, 'q1_purchase_price_percentile', q1_per)

            median_per = float(sum(i < row_participant['median_purchase_price'] for i in self.median_cost_list)) / float(
                len(self.median_cost_list) - 1)
            self.merge_df.set_value(idx, 'median_purchase_price_percentile', median_per)

            q3_per = float(sum(i < row_participant['q3_purchase_price'] for i in self.q3_cost_list)) / float(
                len(self.q3_cost_list) - 1)
            self.merge_df.set_value(idx, 'q3_purchase_price_percentile', q3_per)

            max_per = float(sum(i < row_participant['max_purchase_price'] for i in self.max_cost_list)) / float(
                len(self.max_cost_list) - 1)
            self.merge_df.set_value(idx, 'max_purchase_price_percentile', max_per)
        return

    # a. histogram of common aspect, total and per vertical
    # b. insert aspect per item
    def extract_item_aspect(self):
        # a. histogram of common aspect, total and per vertical
        # self.item_aspects_df = pd.read_csv(self.item_aspects_file)
        return  # TODO add

        aspect_type_dict = self.item_aspects_df['PRDCT_ASPCT_NM'].value_counts().to_dict()
        import operator
        aspect_sort_list = sorted(aspect_type_dict.items(), key=operator.itemgetter(1))  # sort aspect by their common
        aspect_sort_list.reverse()

        sum_total = self.item_aspects_df['PRDCT_ASPCT_NM'].value_counts().sum()
        amount_series_top_K = self.item_aspects_df['PRDCT_ASPCT_NM'].value_counts().nlargest(n=20)
        sum_top_k = amount_series_top_K.sum()
        ratio_remain = float(sum_top_k)/float(sum_total)
        amount_series_top_K.plot.bar(figsize=(8, 6))
        plt.title('TOP 20 item aspect vs. aspect amount, aspect remain: ' + str(round(ratio_remain, 2)))
        plt.ylabel('Amount')
        plt.xlabel('Aspect')
        plt.xticks(rotation=35)
        plot_name = dir_analyze_name + 'top_k_aspects_vs_amount' + '.png'
        plt.savefig(plot_name, bbox_inches='tight')
        # plt.show()
        plt.close()

        # b. insert aspect per item
        # dict key val

        aa = self.item_aspects_df.loc[self.item_aspects_df['PRDCT_ASPCT_NM'] == 'Color']

        a = aa.groupby(['ASPCT_VLU_NM'])
        for buyer_id, group in a:
            print(str(buyer_id) + ': ' + str(group.shape[0]))
        raise
        self.item_buyer_dict = dict(zip(self.purchase_history_df['item_id'], self.purchase_history_df['buyer_id']))
        grouped = self.purchase_history_df.groupby(['buyer_id'])  # groupby how many each user bought

        for buyer_id, group in grouped:
            user_items = group['item_id'].tolist()
            for item_index, item_id in enumerate(user_items):

                cur_item_aspect_df = self.item_aspects_df.loc[self.item_aspects_df['item_id'] == item_id]
                # TODO insert feature of aspects

        return

    # connect to number of purchases
    def insert_purchase_amount_data(self):
        self.merge_df['number_purchase'] = np.nan

        self.user_id_name_dict = dict(zip(self.valid_users_df['USER_ID'], self.valid_users_df.USER_SLCTD_ID))
        self.user_name_id_dict = dict(zip(self.valid_users_df.USER_SLCTD_ID, self.valid_users_df['USER_ID']))
        from math import isnan
        self.user_id_name_dict = {k: self.user_id_name_dict[k] for k in self.user_id_name_dict if not isnan(k)}

        # add number of purchase per user
        sum = 0
        counter_id = 0
        histogram_purchase_list = list()
        grouped = self.purchase_history_df.groupby(['buyer_id'])  # groupby how many each user bought
        for name, group in grouped:

            cur_user_name = self.user_id_name_dict[float(list(group['buyer_id'])[0])]

            # only insert if user in list (74 > 69 ask Hadas)
            if cur_user_name in list(self.merge_df['eBay site user name']):
                cur_idx = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]
                self.merge_df.set_value(cur_idx, 'number_purchase', group.shape[0])
                counter_id += 1
                if group.shape[0] > 200:
                    histogram_purchase_list.append(200)
                else:
                    histogram_purchase_list.append(group.shape[0])
                sum += group.shape[0]

        # calculate purchase threshold
        logging.info('# participant: ' + str(self.merge_df.shape[0]))
        logging.info('# purchases q1: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.25]))
        logging.info('# purchases median: ' + str(self.merge_df['number_purchase'].median()))
        logging.info('# purchases q3: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.75]))

        return self.user_id_name_dict, histogram_purchase_list

    # remove participant with purchase amount below threshold
    # visual purchase histogram
    def slice_participant_using_threshold(self, histogram_purchase_list):
        # remove user buy less than threshold

        self.merge_df = self.merge_df.loc[self.merge_df['number_purchase'] >= self.threshold_purchase]

        logging.info('# participant threshold: ' + str(self.merge_df.shape[0]))
        logging.info('# purchases threshold q1: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.25]))
        logging.info('# purchases threshold median: ' + str(self.merge_df['number_purchase'].median()))
        logging.info('# purchases threshold q3: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.75]))

        self.merge_df.to_csv(self.dir_analyze_name + 'purchase_amount_after_threshold.csv')

        # histogram of number of purchases
        plt.hist(histogram_purchase_list, bins=30)
        plt.title('Histogram of #purchase item per participants')
        plt.ylabel('Participant amount')
        plt.xlabel('#Purchases')
        plot_name = self.dir_analyze_name + 'histogram_purchases_per_user' + '.png'
        plt.savefig(plot_name, bbox_inches='tight')
        # plt.show()
        plt.close()
        return

    # connect to purchase per vertical
    def insert_purchase_vertical_data(self, user_id_name_dict):

        # plot number of purchase per vertical
        vertical_list = list(self.purchase_history_df['BSNS_VRTCL_NAME'].unique())
        amount_series = self.purchase_history_df['BSNS_VRTCL_NAME'].value_counts()
        logging.info('Number of purchases: ' + str(len(self.purchase_history_df['BSNS_VRTCL_NAME'])))
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
            logging.info('Vertical: ' + str(cur_vertical) + ', Amount: ' + str(vertical_amount))
            self.merge_df[str(cur_vertical) + '_amount'] = 0.0
            self.merge_df[str(cur_vertical) + '_ratio'] = 0.0
        # amount and ratio for each vertical
        grouped = self.purchase_history_df.groupby(['buyer_id'])  # groupby how many each user bought
        for name, group in grouped:

            cur_user_name = user_id_name_dict[float(list(group['buyer_id'])[0])]
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
                    self.merge_df.set_value(cur_idx, str(cur_vertical) + '_amount', vec_df_group.shape[0])
                    self.merge_df.set_value(cur_idx, str(cur_vertical) + '_ratio', cur_ratio)

        return

    # calculate pearson correlation for each two features
    def calculate_pearson_correlation(self):

        pearson_f = self.lr_x_feature + self.lr_y_feature       # feature using in calculate pearson
        self.corr_df = self.merge_df[pearson_f]
        corr_df = self.corr_df.corr(method='pearson')
        corr_df.to_csv(dir_analyze_name + 'corr_df.csv')
        corr_abs = corr_df.abs()
        pair_feature_corr = corr_abs.unstack().sort_values(ascending=False).to_dict()

        import operator
        sort_corr_val = sorted(pair_feature_corr.items(), key=operator.itemgetter(1))
        sort_corr_val.reverse()

        dict_already_seen = dict()        # prevent duplicate of feature correlation a_b, b_a

        for idx, corr_obj in enumerate(sort_corr_val):
            feature_a = corr_obj[0][0]
            feature_b = corr_obj[0][1]
            pearson_val = corr_obj[1]
            if pearson_val < 1:

                if pearson_val < self.threshold_pearson:
                    continue

                # check already seen
                if feature_a in dict_already_seen and dict_already_seen[feature_a] == feature_b:
                    continue
                if feature_b in dict_already_seen and dict_already_seen[feature_b] == feature_a:
                    continue

                # check if in same group

                if feature_a in self.time_purchase_meta_feature and feature_b in self.time_purchase_meta_feature:
                    continue
                if feature_a in self.time_purchase_ratio_feature and feature_b in self.time_purchase_ratio_feature:
                    continue
                if feature_a in self.vertical_ratio_feature and feature_b in self.vertical_ratio_feature:
                    continue
                if feature_a in self.purchase_price_feature and feature_b in self.purchase_price_feature:
                    continue
                if feature_a in self.purchase_percentile_feature and feature_b in self.purchase_percentile_feature:
                    continue
                if feature_a in self.user_meta_feature and feature_b in self.user_meta_feature:
                    continue

                logging.info(str(pearson_val) + ' (pearson v.) ' + str(feature_a) + ' vs. ' + str(feature_b))
                dict_already_seen[feature_a] = feature_b

                self.merge_df.plot.scatter(x=feature_a, y=feature_b, figsize=(9, 7))

                z = np.polyfit(self.merge_df[feature_a], self.merge_df[feature_b], 1)
                p = np.poly1d(z)
                plt.plot(self.merge_df[feature_a], p(self.merge_df[feature_a]), "r", linewidth=0.5, alpha=0.5)

                plt.title(feature_a + ' vs. ' + feature_b + ', pearson corr = ' + str(round(pearson_val, 2)))

                plot_name = dir_analyze_name + 'pearson_corr/' + str(round(pearson_val, 2)) + ' pearson,  ' + str(feature_a) + ' vs. ' + str(feature_b) + '.png'

                plt.savefig(plot_name, bbox_inches='tight')

                # plt.show()
                plt.close()
        return

    # calculate linear regression model
    def calculate_linear_regression(self):

        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        for idx, y_feature in enumerate(self.lr_y_feature):
            logging.info('build lr model for: ' + str(y_feature))

            import copy
            relevant_X_columns = copy.deepcopy(self.lr_x_feature)
            print('lr X columns: ' + str(relevant_X_columns))
            if y_feature in relevant_X_columns:
                relevant_X_columns.remove(y_feature)

            bool_percentile = False      # using percentile/real traits values
            if bool_percentile:
                relevant_X_columns.extend(self.trait_percentile)
                raise
            else:
                cur_all_columns = relevant_X_columns + [self.lr_y_feature]

            self.raw_df = self.merge_df.copy(deep=True)     # self.merge_df[relevant_X_columns]
            self.raw_df.to_csv(self.dir_analyze_name + 'raw_df_before_split.csv')

            bool_split = False
            if bool_split:
                train_df, test_df = train_test_split(self.merge_df, test_size=self.test_fraction)
                X_train = train_df[relevant_X_columns]
                y_train = train_df[y_feature]
                X_test = test_df[relevant_X_columns]
                y_test = test_df[y_feature]
            else:
                X_train = self.merge_df[relevant_X_columns]
                y_train = self.merge_df[y_feature]

            import statsmodels.api as sm
            from scipy import stats

            X_train_final = sm.add_constant(X_train)
            est = sm.OLS(y_train, X_train_final)
            est2 = est.fit()
            print(est2.summary())
            dict_f_p = dict(est2.pvalues)
            dict_pa_pv = dict(est2.params)

            import operator
            sorted_p_val = sorted(dict_f_p.items(), key=operator.itemgetter(1))

            print('Traits name: ' + str(y_feature))
            print('R_square: ' + str(est2.rsquared))
            line_new = '%35s  %15s  %15s' % ('Feature name', 'Coeff', 'P Value')

            print()
            print()
            print()
            print(line_new)
            for c_tuple in sorted_p_val:
                f_name = c_tuple[0]
                f_pval = c_tuple[1]
                f_coeff = dict_pa_pv[f_name]
                # print(str(f_name.ljust(35)) + str(f_coeff.ljust(15)) + str(f_pval.ljust(15)))
                # print('{:>35} {:>15} {:>15}'.format([f_name, f_coeff, f_pval]))
                line_new = '%35s  %15s  %15s' % (f_name, round(f_coeff, 3), round(f_pval, 4))
                print(line_new)
            print()
            raise
            r_square = est2.rsquared()
            dict(est2.pvalues)

            table = est2.summary()
            p1 = fit.pvalues[i]

            # X_train = self.merge_df[relevant_X_columns]
            # y_train = self.merge_df[y_feature]

            # Create linear regression object

            # regr = linear_model.LinearRegression()
            for i in range(1, 20):
                regr = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=i)
            # regr = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=6)
            # Train the model using the training sets
                regr.fit(X_train, y_train)
                print('train score -  ' + str(i) + ',  r2: ' + str(regr.score(X_train, y_train)))
            for i in range(1, 20):
                regr = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=i)
            # regr = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=6)
            # Train the model using the training sets
                regr.fit(X_train, y_train)
                print('test score -  ' + str(i) + ',  r2: ' + str(regr.score(X_test, y_test)))

            # viaual regression model
            # Make predictions using the testing set

            import statsmodels.api as sm
            from scipy import stats

            X_train_final = sm.add_constant(X_train)
            est = sm.OLS(y_train, X_train_final)
            est2 = est.fit()
            print(est2.summary())
            table = est2.summary()
            p1 = fit.pvalues[i]

            X = X_train_final
            y = y_train

            from scipy import stats
            import numpy as np
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            print(slope)
            print(intercept)
            print(p_value)
            raise

            import pandas as pd
            import numpy as np
            from sklearn import datasets, linear_model
            from sklearn.linear_model import LinearRegression
            import statsmodels.api as sm
            from scipy import stats

            lm = linear_model.LinearRegression()
            lm.fit(X, y)
            params = np.append(lm.intercept_, lm.coef_)
            predictions = lm.predict(X)

            newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
            MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

            # Note if you don't want to use a DataFrame replace the two lines above with
            # newX = np.append(np.ones((len(X),1)), X, axis=1)
            # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

            var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
            sd_b = np.sqrt(var_b)
            ts_b = params / sd_b

            p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

            sd_b = np.round(sd_b, 3)
            ts_b = np.round(ts_b, 3)
            p_values = np.round(p_values, 3)
            params = np.round(params, 4)

            myDF3 = pd.DataFrame()
            myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b,ts_b, p_values]
            print(myDF3)

            raise




            regr = linear_model.LinearRegression()

            # regr = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=10)
            regr.fit(X_train, y_train)
            regr.score(X_train, y_train)
            y_pred = regr.predict(X_train)

            # The coefficients
            cur_r2 = r2_score(y_train, y_pred)

            logging.info('Coefficients: \n', regr.coef_)
            # The mean squared error
            logging.info("Mean squared error: %.2f"
                  % mean_squared_error(y_train, y_pred))
            # Explained variance score: 1 is perfect prediction
            logging.info('Variance score: %.2f' % cur_r2)

            bool_trend_line = False
            if bool_trend_line:
                pred_train_dict = dict(zip(y_pred, y_train))
                import collections
                od = collections.OrderedDict(sorted(pred_train_dict.items()))
                od_y_train = od.values()
                od_y_pred = od.keys()
                import numpy as np

                z = np.polyfit(od_y_train, od_y_pred, 1)
                p = np.poly1d(z)
                plt.plot(od_y_train, p(od_y_pred), "r--")

            # Plot outputs - train
            plt.scatter(y_train, y_pred, color='black')
            amount_data = len(y_pred)
            plt.xlim(0, 5)
            plt.ylim(0, 5)
            plt.xlabel('Real')
            plt.ylabel('Prediction')

            plt.title('Train: ' + str(y_feature) + ', R2: ' + str(round(cur_r2, 2)) + ', Amount: ' + str(amount_data))
            plt.show()
            plt.close()

            # Plot outputs
            y_pred_test = regr.predict(X_test)
            # The coefficients
            cur_r2_test = r2_score(y_test, y_pred_test)

            plt.scatter(y_test, y_pred_test, color='black')
            amount_data_test = len(y_pred_test)
            plt.xlim(0, 5)
            plt.ylim(0, 5)
            plt.xlabel('Real')
            plt.ylabel('Prediction')

            plt.title('Test: ' + str(y_feature) + ', R2: ' + str(round(cur_r2_test, 2)) + ', Amount: ' + str(amount_data_test))
            plt.show()
            plt.close()

            plt.figure(1, figsize = (6, 8))
            plt.title('Openness trait')
            plt.subplot(211)
            plt.scatter(y_train, y_pred, color='black')
            amount_data = len(y_pred)
            plt.xlim(0, 5)
            plt.ylim(0, 5)
            # plt.xlabel('Real')
            plt.ylabel('Prediction')
            plt.title('Train: ' + str(y_feature) + ', R2: ' + str(round(cur_r2, 2)) + ', Amount: ' + str(amount_data))

            plt.subplot(212)
            plt.scatter(y_test, y_pred_test, color='black')
            amount_data_test = len(y_pred_test)
            plt.xlim(0, 5)
            plt.ylim(0, 5)
            plt.xlabel('Real')
            plt.ylabel('Prediction')

            plt.title('Test: ' + str(y_feature) + ', R2: ' + str(round(cur_r2_test, 2)) + ', Amount: ' + str(
                amount_data_test))


            # plt.show()

            plot_name = self.dir_analyze_name + '/regression_result/' + str(round(cur_r2_test, 2)) + '_' + \
                        str(round(cur_r2, 2)) + '_' + str(y_feature) + '.png'

            plt.savefig(plot_name, bbox_inches='tight')

            plt.close()



            # statistics data

            import pandas as pd
            import numpy as np
            from sklearn import datasets, linear_model
            from sklearn.linear_model import LinearRegression
            import statsmodels.api as sm
            from scipy import stats

            diabetes = datasets.load_diabetes()
            X = X_train
            y = y_train

            X2 = sm.add_constant(X)
            est = sm.OLS(y, X2)
            est2 = est.fit()
            print(est2.summary())

            print('s')

        return

    # calculate logistic regression model
    def calculate_logistic_regression(self):

        # self.map_dict_percentile_group = dict(zip(self.lr_y_logistic_feature,  self.trait_percentile))

        # contain test results data for all the iterations
        true_list = list()
        false_list = list()
        test_score = list()

        train_score_check = list()
        train_score_check2 = list()
        test_score_check = list()
        c_check = list()

        # test score for each trait
        openness_score = list()
        conscientiousness_score = list()
        extraversion_score = list()
        agreeableness_score = list()
        neuroticism_score = list()

        import copy
        relevant_X_columns = copy.deepcopy(self.lr_x_feature)
        map_dict_feature_non_zero = dict()
        for trait in self.lr_y_logistic_feature:
            map_dict_feature_non_zero[trait] = dict(zip(list(relevant_X_columns), [0]*len(relevant_X_columns)))

        # add column H/L for each trait
        self.add_high_low_traits_column()

        from sklearn import linear_model
        from sklearn.model_selection import train_test_split

        for i in range(0, 50):  # iterate N iterations
            for idx, y_feature in enumerate(self.lr_y_logistic_feature):    # iterate each trait
                logging.info('build lr model for: ' + str(y_feature))

                # create gap (throw near median results)
                if self.bool_slice_gap_percentile:
                    cur_f = self.map_dict_percentile_group[y_feature]
                    self.merge_df.to_csv(self.dir_analyze_name + 'logistic_regression_merge_df.csv')
                    h_df = self.merge_df.loc[self.merge_df[cur_f] >= self.h_limit]
                    l_df = self.merge_df.loc[self.merge_df[cur_f] <= self.l_limit]
                    frames = [l_df, h_df]
                    self.raw_df = pd.concat(frames)
                else:
                    self.raw_df = self.merge_df

                self.merge_df.to_csv(self.dir_analyze_name + 'participant_threshold_20_features_extraction.csv')

                import copy
                relevant_X_columns = copy.deepcopy(self.lr_x_feature)
                if y_feature in relevant_X_columns:
                    relevant_X_columns.remove(y_feature)

                self.raw_df = self.raw_df[relevant_X_columns + [y_feature]]

                if self.bool_normalize_features:
                    self.raw_df = self.preprocessing_min_max(self.raw_df)

                train_df, test_df = train_test_split(self.raw_df, test_size=self.test_fraction)
                X_train = train_df[relevant_X_columns]
                y_train = train_df[y_feature]
                X_test = test_df[relevant_X_columns]
                y_test = test_df[y_feature]
                X_train.to_csv(self.dir_analyze_name + 'logistic_regression_extraversion_df.csv')
                y_train.to_csv(self.dir_analyze_name + 'logistic_regression_y_df.csv')

                from sklearn import model_selection
                regr = linear_model.LogisticRegressionCV(Cs=[4.5],
                                                         penalty=self.penalty,
                                                         solver='liblinear',
                                                         cv=model_selection.StratifiedKFold(n_splits=4,
                                                                                             shuffle=True,
                                                                                             random_state=None))

                regr.fit(X_train, y_train)
                train_score = regr.score(X_train, y_train)
                test_score = regr.score(X_test, y_test)

                print('test_score: ' + str(test_score))
                print('train_score: ' + str(train_score))
                train_score_check.append(train_score)
                test_score_check.append(test_score)
                c_check.append(regr.C_[0])

                map_dict_feature_coeff = dict(zip(list(relevant_X_columns), regr.coef_[0]))
                import operator
                sorted_x = sorted(map_dict_feature_coeff.items(), key=operator.itemgetter(1))
                sorted_x.reverse()
                print(y_feature)
                for tuple in sorted_x:
                    print(str(tuple[0]) + ': ' + str(tuple[1]))
                print('')

                '''
                check CV logistic regression using best C found
                regr2 = linear_model.LogisticRegression(C=regr.C_[0],  # Cs=[1, 5, 10],
                    penalty=self.penalty,
                    solver='liblinear')
                # regr = linear_model.LogisticRegression(C=self.C, penalty=self.penalty, solver='liblinear')
                regr2.fit(X_train, y_train)
                train_score = regr.score(X_train, y_train)
                train_score_check2.append(train_score)'''
                #import eli5

                #print(eli5.lime(clf=regr))
                # print(a)
                # b = eli5.explain_prediction_sklearn(regr, X_train)
                # print('b')
                # regr = linear_model.LogisticRegression(C=self.C, penalty=self.penalty)
                # regr.fit(X_train, y_train)

                # train_score = regr.score(X_train, y_train)
                # test_score = regr.score(X_test, y_test)

                # print('Current target feature: ' + str(y_feature))
                # print('train score: ' + str(train_score) + ', #samples: ' + str(len(y_train)))
                # print('test score: ' + str(test_score) + ', #samples: ' + str(len(y_test)))

                if y_feature == 'openness_group':
                    openness_score.append(test_score)
                if y_feature == 'conscientiousness_group':
                    conscientiousness_score.append(test_score)
                if y_feature == 'extraversion_group':
                    extraversion_score.append(test_score)
                if y_feature == 'agreeableness_group':
                    agreeableness_score.append(test_score)
                if y_feature == 'neuroticism_group':
                    neuroticism_score.append(test_score)

                for idx, pair_prob in enumerate(regr.predict_proba(X_test)):
                    true_label = list(y_test)[idx]
                    max(pair_prob)
                    if pair_prob[0] > pair_prob[1]:
                        pred_label = 0
                    else:
                        pred_label = 1

                    if true_label == pred_label:
                        # true_list.append(abs(0.5-pair_prob[pred_label]))
                        true_list.append(pair_prob[pred_label])
                    else:
                        # false_list.append(abs(0.5-pair_prob[pred_label]))
                        false_list.append(pair_prob[pred_label])


        # average of all iterations

        '''print('Test openness_trait: ' + str(sum(openness_score) / len(openness_score)))
        print('Test conscientiousness_trait: ' + str(sum(conscientiousness_score) / len(conscientiousness_score)))
        print('Test extraversion_trait: ' + str(sum(extraversion_score) / len(extraversion_score)))
        print('Test agreeableness_score: ' + str(sum(agreeableness_score) / len(agreeableness_score)))
        print('Test neuroticism_score: ' + str(sum(neuroticism_score) / len(neuroticism_score)))'''

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

        print('test score: ' + str(sum(test_score_check) / len(test_score_check)))
        print('train score: ' + str(sum(train_score_check) / len(train_score_check)))
        # print('train score2: ' + str(sum(train_score_check2) / len(train_score_check2)))
        print('C values: ' + str(c_check))
        from collections import Counter

        print('Counter C: ' + str(Counter(c_check)))

        '''print(true_list)
        print(len(true_list))
        print(false_list)
        print(len(false_list))'''

        # print('total ratio: ' + str(float(len(true_list))/float(len(true_list)+len(false_list))))
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
        return

    # normalized each column seperatly between zero to one
    def preprocessing_min_max(self, df):

        from sklearn import preprocessing
        norm_method = 'min_max'

        if norm_method == 'min_max':
            normalized_df = (df - df.min()) / (df.max() - df.min())
        elif norm_method == 'mean_normalization':
            normalized_df = (df-df.mean())/df.std()
        else:
            raise('undefined norm method')

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
            logging.info('Change column values (reverse mode): ' + str(filter_col))
            self.participant_df[filter_col] = self.participant_df[filter_col].apply(lambda x: 6 - x)
        return

    # calculate traits valuers and percentile per participant
    def cal_participant_traits_values(self):

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

        # add average traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate traits value for participant: ' + str(row_participant['Email address']))
            self.calculate_individual_score(idx, row_participant)

        # add percentile traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self.cal_participant_traits_percentile_values(idx, row_participant)

        # after calculate traits score+percentile extract only relevant features
        '''remain_feature_list = ['Full Name', 'Gender', 'eBay site user name', 'Age', 'openness_trait',
                               'conscientiousness_trait', 'extraversion_trait', 'agreeableness_trait',
                               'neuroticism_trait', 'openness_percentile', 'conscientiousness_percentile',
                               'extraversion_percentile', 'agreeableness_percentile', 'neuroticism_percentile',
                               'age_group']'''

        self.merge_df = self.participant_df.copy()

        # self.merge_df = self.participant_df[remain_feature_list].copy()

        return

    # after delete un valid participant
    def cal_all_participant_percentile_value(self):
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self.cal_participant_traits_percentile_values(idx, row_participant)

            # after calculate traits score+percentile extract only relevant features
        remain_feature_list = ['Full Name', 'Gender', 'eBay site user name', 'Age', 'openness_trait',
                               'conscientiousness_trait', 'extraversion_trait', 'agreeableness_trait',
                               'neuroticism_trait', 'openness_percentile', 'conscientiousness_percentile',
                               'extraversion_percentile', 'agreeableness_percentile', 'neuroticism_percentile',
                               'age_group']

        # self.merge_df = self.merge_df[remain_feature_list].copy()
        return

    # calculate traits values for one participant
    def calculate_individual_score(self, idx, row_participant):

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
    def cal_participant_traits_percentile_values(self, idx, participant_score):

        op_per = float(sum(i < participant_score['openness_trait'] for i in self.openness_score_list))/float(len(self.openness_score_list)-1)
        self.participant_df.set_value(idx, 'openness_percentile', op_per)
        co_per = float(sum(
            i < participant_score['conscientiousness_trait'] for i in self.conscientiousness_score_list))/float(len(self.conscientiousness_score_list)-1)
        self.participant_df.set_value(idx, 'conscientiousness_percentile', co_per)
        ex_per = float(sum(
            i < participant_score['extraversion_trait'] for i in self.extraversion_score_list))/float(len(self.extraversion_score_list)-1)
        self.participant_df.set_value(idx, 'extraversion_percentile', ex_per)
        ag_per = float(sum(
            i < participant_score['agreeableness_trait'] for i in self.agreeableness_score_list))/float(len(self.agreeableness_score_list)-1)
        self.participant_df.set_value(idx, 'agreeableness_percentile', ag_per)
        ne_per = float(sum(
            i < participant_score['neuroticism_trait'] for i in self.neuroticism_score_list))/float(len(self.neuroticism_score_list)-1)
        self.participant_df.set_value(idx, 'neuroticism_percentile', ne_per)

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

    calculate_obj.init_debug_log()                      # init log file
    calculate_obj.load_clean_csv_results()              # load data set
    calculate_obj.clean_df()                            # clean df - e.g. remain valid users only
    calculate_obj.create_feature_list()                 # create x_feature

    # calculate personality trait per user + percentile per trait
    calculate_obj.change_reverse_value()                # change specific column into reverse mode
    calculate_obj.cal_participant_traits_values()       # calculate average traits and percentile value
    calculate_obj.insert_gender_feature()               # add gender feature

    calculate_obj.extract_user_purchase_connection()    # insert purchase and vertical type to model
    calculate_obj.insert_money_feature()                # add feature contain money issue
    calculate_obj.insert_time_feature()                 # add time purchase feature

    # calculate_obj.textual_feature()                       # TODO even basic feature
    # calculate_obj.extract_item_aspect()                   # TODO add
    # calculate_obj.cal_all_participant_percentile_value()  # TODO need to add

    calculate_obj.calculate_logistic_regression()       # predict traits H or L
    # calculate_obj.calculate_linear_regression()         # predict traits using other feature
    # calculate_obj.calculate_pearson_correlation()       # calculate pearson correlation

if __name__ == '__main__':

    # input file name
    participant_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_139_participant.csv'
    item_aspects_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_item_aspects.csv'
    purchase_history_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_purchase_history.csv'
    valid_users_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_valid_users.csv'
    dir_analyze_name = '/Users/sguyelad/PycharmProjects/research/survey_pilot/analyze_pic/'
    threshold_purchase = 30
    c_value = 2
    regularization = 'l1'  # 'l2'
    bool_slice_gap_percentile = True
    bool_normalize_features = True

    main(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
         threshold_purchase, bool_slice_gap_percentile, bool_normalize_features, c_value, regularization)
