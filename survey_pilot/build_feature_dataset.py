from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

class CalculateScore:

    def __init__(self, participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
                 threshold_purchase, bool_slice_gap_percentile=True, bool_normalize_features=True, cur_C=2,
                 cur_penalty='l1', time_purchase_ratio_feature_flag=True, time_purchase_meta_feature_flag=True,
                 vertical_ratio_feature_flag=True, purchase_percentile_feature_flag=True,
                 user_meta_feature_flag=True, aspect_feature_flag=True, h_limit=0.6, l_limit=0.4,
                 k_best=10, plot_directory='', user_type='all', normalize_traits=True):

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

        self.global_test = 0.0
        self.global_train_cv = 0.0
        self.global_counter = 0.0

        self.question_openness = [5, 10, 15, 20, 25, 30, 35, 40, 41, 44]
        self.question_conscientiousness = [3, 8, 13, 18, 23, 28, 33, 43]
        self.question_extraversion = [1, 6, 11, 16, 21, 26, 31, 36]
        self.question_agreeableness = [2, 7, 12, 17, 22, 27, 32, 37, 42]
        self.question_neuroticism = [4, 9, 14, 19, 24, 29, 34, 39]

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.verbose_flag = True

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
        self.k_best = k_best                        # number of k_best feature to select
        self.plot_directory = plot_directory
        self.user_type = user_type                  # user type in model 'all'/'cf'/'ebay-tech'
        self.normalize_traits = normalize_traits    # normalize each trait to 0-1

        self.pearson_relevant_feature = ['Age', 'openness_percentile',
                   'conscientiousness_percentile', 'extraversion_percentile', 'agreeableness_percentile',
                   'neuroticism_percentile', 'number_purchase', 'Electronics_ratio', 'Fashion_ratio',
                   'Home & Garden_ratio', 'Collectibles_ratio', 'Lifestyle_ratio', 'Parts & Accessories_ratio',
                   'Business & Industrial_ratio', 'Media_ratio']

        self.lr_x_feature = list()

        self.lr_y_feature = ['agreeableness_trait', 'extraversion_trait', 'neuroticism_trait', 'conscientiousness_trait', 'openness_trait']

        # traits to check
        self.lr_y_logistic_feature = ['openness_group', 'conscientiousness_group', 'extraversion_group','agreeableness_group', 'neuroticism_group']
        # self.lr_y_logistic_feature = ['openness_group']

        # trait to predict in regression model
        self.lr_y_linear_feature = ['openness_group', 'conscientiousness_group', 'extraversion_group',
                                      'agreeableness_group', 'neuroticism_group']
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
        self.aspect_feature_flag = aspect_feature_flag

        self.time_purchase_ratio_feature = ['day_ratio', 'evening_ratio', 'night_ratio', 'weekend_ratio']
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

        self.aspect_feature = ['color_ratio', 'colorful_ratio', 'protection_ratio', 'country_ratio', 'brand_ratio',
                               'brand_unlabeled_ratio']

        self.logistic_regression_accuracy = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

        self.logistic_regression_roc = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

        self.logistic_regression_accuracy_cv = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

        self.linear_regression_mae = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

        self.linear_regression_pearson = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

    def init_debug_log(self):
        import logging

        lod_file_name = '/Users/gelad/Personality-based-commerce/BFI_results/log/' + 'build_feature_dataset_' + str(self.cur_time) + '.log'

        # logging.getLogger().addHandler(logging.StreamHandler())

        logging.basicConfig(filename=lod_file_name,
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        # print result in addition to log file
        if self.verbose_flag:
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

        logging.info("")
        logging.info("")
        logging.info("start log program")

        return

    # load csv into df
    def load_clean_csv_results(self):

        self.participant_df = pd.read_csv(self.participant_file)
        self.item_aspects_df = pd.read_csv(self.item_aspects_file)
        self.purchase_history_df = pd.read_csv(self.purchase_history_file)
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

        return

    def insert_gender_feature(self):

        self.merge_df = self.participant_df.copy()

        self.merge_df['gender'] = \
            np.where(self.merge_df['Gender'] == 'Male', 1, 0)

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_gender.csv')
        logging.info('')
        logging.info('add gender feature')
        logging.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df_gender.csv')
        return

    # remain users regards to user_type variable (eBay/tech/CF)
    def remove_except_cf(self):

        logging.info('')
        logging.info('extract user regards to user_type variable ' + str(self.user_type))

        if self.user_type not in ['all', 'cf', 'ebay-tech']:
            raise('undefined user_type: ' + str(self.user_type))

        if self.user_type == 'cf':
            logging.info('Remain only users from CF')
            self.merge_df = self.merge_df.loc[self.merge_df['Site'] == 'CF']
            logging.info('CF users: ' + str(self.merge_df.shape[0]))
        elif self.user_type == 'ebay-tech':
            logging.info('Remain only users from eBay and Tech')
            self.merge_df = self.merge_df.loc[self.merge_df['Site'] != 'CF']
            logging.info('Is users: ' + str(self.merge_df.shape[0]))
        elif self.user_type == 'all':
            logging.info('Remain all users')
            logging.info('Is users: ' + str(self.merge_df.shape[0]))
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
        logging.info('')
        logging.info('add user purchase connection')
        logging.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df.csv')
        return

    # remove participant with purchase amount below threshold
    # visual purchase histogram
    def _slice_participant_using_threshold(self, histogram_purchase_list):
        # remove user buy less than threshold
        before_slice_users = self.merge_df.shape[0]
        self.merge_df = self.merge_df.loc[self.merge_df['number_purchase'] >= self.threshold_purchase]
        logging.info('')
        logging.info('Threshold used: ' + str(self.threshold_purchase))
        logging.info('# participant after slice threshold: ' + str(self.merge_df.shape[0]))
        logging.info('# participant deleted: ' + str(before_slice_users - self.merge_df.shape[0]))
        logging.info('# purchases threshold q1: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.25]))
        logging.info('# purchases threshold median: ' + str(self.merge_df['number_purchase'].median()))
        logging.info('# purchases threshold q3: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.75]))

        self.merge_df.to_csv(self.dir_analyze_name + 'purchase_amount_after_threshold.csv')
        logging.info('')
        logging.info('slice particpant below purchase threshold')
        logging.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'purchase_amount_after_threshold.csv')

        # histogram of number of purchases
        plt.hist(histogram_purchase_list, bins=30)
        plt.title('Histogram of #purchase item per participants, #P ' + str(self.merge_df.shape[0]))
        plt.ylabel('Participant amount')
        plt.xlabel('#Purchases')
        plot_name = self.dir_analyze_name + 'histogram_purchases_per_user' + '_p_' + str(self.merge_df.shape[0]) + '_threshold_' + str(self.threshold_purchase) + '.png'
        plt.savefig(plot_name, bbox_inches='tight')
        # plt.show()
        plt.close()
        return

    # connect to purchase per vertical
    def _insert_purchase_vertical_data(self, user_id_name_dict):

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

        logging.info('')
        logging.info('start to extract item aspect feature')

        from build_item_aspect_feature import BuildItemAspectScore

        item_aspect_obj = BuildItemAspectScore(self.item_aspects_df, self.participant_df, self.purchase_history_df,
                                               self.valid_users_df, self.merge_df, self.user_id_name_dict, self.aspect_feature)
        item_aspect_obj.add_aspect_features()
        self.merge_df = item_aspect_obj.merge_df
        logging.info('number of features after add item aspect: ' + str(self.merge_df.shape[1]))

        # self.corr_df = self.merge_df[pearson_f]
        # corr_df = self.merge_df.corr(method='pearson')
        # corr_df.to_csv(self.dir_analyze_name + 'corr_df_item_aspect.csv')
        # a = 5
        '''
        print common item aspect histogram
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
        # plot_name = dir_analyze_name + 'top_k_aspects_vs_amount' + '.png'
        # plt.savefig(plot_name, bbox_inches='tight')
        plt.show()
        plt.close()'''

        # b. insert aspect per item
        # dict key val

        '''aa = self.item_aspects_df.loc[self.item_aspects_df['PRDCT_ASPCT_NM'] == 'Color']

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
                # TODO insert feature of aspects'''

        return

    # TODO edit - add textual feature
    def textual_feature(self):
        # self.merge_df['title_length'] = np.nan
        return

    # normalize trait to 0-1 scale (div by 5)
    def normalize_personality_trait(self):

        logging.info('')
        logging.info('normalize flag: ' + str(self.normalize_traits))
        if self.normalize_traits:
            for c_trait in self.lr_y_feature:
                self.merge_df[c_trait] = self.merge_df[c_trait] / 5.0
                logging.info('finish normalize trait: ' + str(c_trait))
                logging.info('Average trait: ' + str(self.merge_df[c_trait].mean()))
                logging.info('Std trait: ' + str(self.merge_df[c_trait].std()))

        return

    # calculate traits valuers and percentile per participant
    def cal_participant_traits_values(self):

        # add personality traits empty columns
        self._add_traits_feature_columns()

        # add average traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate traits value for participant: ' + str(row_participant['Email address']))
            self._calculate_individual_score(idx, row_participant)

        # add percentile traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            # logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
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

        logging.info('')
        logging.info('extract money features')

        self._add_price_feature_columns()
        self._add_price_feature()                # insert value feature
        self._add_percentile_price_feature()     # insert percentile feature

        self.merge_df.to_csv(self.dir_analyze_name + 'merge_df_cost_value_percentile.csv')
        logging.info('')
        logging.info('add cost value percentile features')
        logging.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df_cost_value_percentile.csv')
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

                time_object = time.strptime(cur_per, '%d/%m/%Y %H:%M')
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
        logging.info('')
        logging.info('add time purchase features')
        logging.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'merge_df_time_purchase.csv')
        return

    # mapping of user country and time zone
    @staticmethod
    def _find_time_zone_shift(country):

        def find_shift(country):        # relative to -7 (server in USA)
            return {
                'Israel': 10,
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
                'Argentina': 3
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
            raise('unknown key type: self.user_id_name_dict.keys()[0]')

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
        logging.info('# participant: ' + str(self.merge_df.shape[0]))
        logging.info('# purchases q1: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.25]))
        logging.info('# purchases median: ' + str(self.merge_df['number_purchase'].median()))
        logging.info('# purchases q3: ' + str(self.merge_df['number_purchase'].quantile([.25, .5, .75])[0.75]))

        return self.user_id_name_dict, histogram_purchase_list

    # calculate pearson correlation for each two features
    def calculate_pearson_correlation_old(self):

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

    def calculate_pearson_correlation(self, relevant_X_columns, y_feature, df):

        # for idx, y_feature in enumerate(self.lr_y_feature):  # iterate each trait
        pearson_f = relevant_X_columns + self.lr_y_feature     # feature using in calculate pearson
        self.corr_df = self.merge_df[pearson_f]
        corr_df = self.corr_df.corr(method='pearson')
        corr_df.to_csv(self.dir_analyze_name + 'corr_df.csv')
        logging.info('')
        logging.info('save pearson correlation df')
        logging.info('Save file: self.corr_df - ' + str(self.dir_analyze_name) + 'corr_df.csv')
        print(corr_df)
        return

    # calculate logistic regression model
    def calculate_logistic_regression(self):

        # self.map_dict_percentile_group = dict(zip(self.lr_y_logistic_feature,  self.trait_percentile))

        # contain test results data for all the iterations
        true_list = list()
        false_list = list()
        test_score = list()
        train_score_check = list()
        test_score_check = list()

        # test score for each trait
        openness_score = list()
        conscientiousness_score = list()
        extraversion_score = list()
        agreeableness_score = list()
        neuroticism_score = list()

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
        map_dict_feature_non_zero = dict()
        for trait in self.lr_y_logistic_feature:
            map_dict_feature_non_zero[trait] = dict(zip(list(relevant_X_columns), [0]*len(relevant_X_columns)))

        # add column H/L for each trait
        self.add_high_low_traits_column()

        from sklearn import linear_model
        from sklearn import model_selection
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        from sklearn.feature_selection import SelectKBest

        for i in range(0, 1):  # iterate N iterations
            for idx, y_feature in enumerate(self.lr_y_logistic_feature):    # build model for each trait separately
                logging.info('')
                logging.info('build lr model for: ' + str(y_feature))

                self.split_bool = False

                # create gap (throw near median results)
                if self.bool_slice_gap_percentile:
                    cur_f = self.map_dict_percentile_group[y_feature]
                    self.merge_df.to_csv(self.dir_analyze_name + 'logistic_regression_merge_df.csv')

                    '''
                    # distribution of percentile users
                    data = self.merge_df[cur_f].tolist()
                    plt.hist(data, bins=20, alpha=0.5)
                    plt.title(str(cur_f))
                    plt.xlabel('percentile')
                    plt.ylabel('amount')

                    plt.show()
                    
                    print(y_feature)
                    print(self.threshold_purchase)
                    print(self.merge_df.shape[0])
                    print('80: ' + str(self.merge_df.loc[self.merge_df[cur_f] >= 0.8].shape[0]))
                    print('70: ' + str(self.merge_df.loc[self.merge_df[cur_f] >= 0.7].shape[0]))
                    print('60: ' + str(self.merge_df.loc[self.merge_df[cur_f] >= 0.6].shape[0]))

                    print('40: ' + str(self.merge_df.loc[self.merge_df[cur_f] <= 0.4].shape[0]))
                    print('30: ' + str(self.merge_df.loc[self.merge_df[cur_f] <= 0.3].shape[0]))
                    print('20: ' + str(self.merge_df.loc[self.merge_df[cur_f] <= 0.2].shape[0]))

                    print('80-20: ' + str(self.merge_df.loc[self.merge_df[cur_f] >= 0.8].shape[0] + self.merge_df.loc[self.merge_df[cur_f] <= 0.2].shape[0]))
                    print('70-30: ' + str(self.merge_df.loc[self.merge_df[cur_f] >= 0.7].shape[0] + self.merge_df.loc[self.merge_df[cur_f] <= 0.3].shape[0]))
                    print('60-40: ' + str(self.merge_df.loc[self.merge_df[cur_f] >= 0.6].shape[0] +  self.merge_df.loc[self.merge_df[cur_f] <= 0.4].shape[0]))
                    continue
                    '''
                    h_df = self.merge_df.loc[self.merge_df[cur_f] >= self.h_limit]
                    l_df = self.merge_df.loc[self.merge_df[cur_f] <= self.l_limit]

                    logging.info('H group amount: ' + str(h_df.shape[0]))
                    logging.info('L group amount: ' + str(l_df.shape[0]))

                    frames = [l_df, h_df]
                    self.raw_df = pd.concat(frames)
                else:
                    self.raw_df = self.merge_df

                self.raw_df = self.raw_df.fillna(0)
                self.raw_df.to_csv(self.dir_analyze_name + 'lr_final_data.csv')
                logging.info('')
                logging.info('Save file: self.raw_df - ' + str(self.dir_analyze_name) + 'lr_final_data.csv')

                import copy
                relevant_X_columns = copy.deepcopy(self.lr_x_feature)
                if y_feature in relevant_X_columns:
                    relevant_X_columns.remove(y_feature)

                self.raw_df = self.raw_df[relevant_X_columns + [y_feature]]

                if self.bool_normalize_features:
                    self.raw_df = self.preprocessing_min_max(self.raw_df)

                X = self.raw_df[relevant_X_columns]
                y = self.raw_df[y_feature]

                if self.split_bool:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        stratify=y,
                        test_size=self.test_fraction
                    )
                    logging.info('train: class 0 ratio: ' + str(sum(y_train) / len(y_train)))
                    logging.info('test: class 0 ratio: ' + str(sum(y_test) / len(y_test)))
                else:
                    X_train = X
                    y_train = y

                logging.info('all: class 0 ratio:  ' + str(sum(y)/len(y)))

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
                from sklearn.feature_selection import f_classif
                X_train.to_csv(self.dir_analyze_name + 'X_train_check.csv')
                # X_test.to_csv(self.dir_analyze_name + 'X_test_check.csv')
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

                logging.info('Total sample size: ' + str(self.raw_df.shape[0]))
                logging.info('Number of features before selecting: ' + str(self.raw_df.shape[1]))
                logging.info('Number of k best features: ' + str(X_train.shape[1]))

                # regr = linear_model.LogisticRegression(penalty=self.penalty, C=self.C)
                regr = linear_model.LogisticRegressionCV(# Cs=1,
                                                         penalty=self.penalty,
                                                         solver='liblinear',
                    cv=model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=None))

                '''cv=model_selection.StratifiedKFold(n_splits=4,shuffle=True,random_state=None))'''

                regr.fit(X_train, y_train)
                # train_score = regr.score(X_train, y_train)
                if self.split_bool:
                    test_score = regr.score(X_test, y_test)
                    prob_test_score = regr.predict_proba(X_test)
                    y_1_prob = prob_test_score[:,1]
                    roc_score = roc_auc_score(y_test, y_1_prob)     # macro roc score
                    fpr, tpr, _ = roc_curve(y_test, y_1_prob)
                    auc_score = auc(fpr, tpr)

                c_index = np.where(regr.Cs_ == regr.C_[0])[0][0]
                train_score = sum(regr.scores_[1][:, c_index]) / 4  # num splits

                # self.global_test += test_score
                self.global_train_cv += train_score
                self.global_counter += 1

                if self.global_counter % 100 == 0:
                    print('Counter: ' + str(self.global_counter))
                    # print('Test Avg: ' + str(self.global_test/self.global_counter))
                    print('Train Avg: ' + str(self.global_train_cv / self.global_counter))

                if self.split_bool:
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange',
                             lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(str(y_feature) + ': test amount ' + str(X_test.shape[0]) + ', test prop ' + str(round(sum(y_test) / len(y_test),2)))
                    plt.legend(loc="lower right")

                    plot_name = str(round(auc_score, 2)) + '_ROC_k=' + str(self.k_best) + '_penalty=' + str(
                        self.penalty) + '_gap=' + str(
                        self.h_limit) + '_' + str(self.l_limit) + '_test_amount=' + \
                                str(X_test.shape[0]) + '_threshold=' + str(self.threshold_purchase) + '_trait=' + str(y_feature) + '_max=' + str(round(auc_score, 2)) + '.png'

                    import os
                    if not os.path.exists(self.plot_directory + '/roc/'):
                        os.makedirs(self.plot_directory + '/roc/')

                    plot_path = self.plot_directory + '/roc/' + plot_name
                    plt.savefig(plot_path, bbox_inches='tight')

                    plt.close()

                logging.info("")
                logging.info('train_score: ' + str(train_score))

                if self.split_bool:
                    logging.info('test_score: ' + str(test_score))
                    logging.info('roc score: ' + str(roc_score))
                    logging.info('auc score: ' + str(auc_score))
                logging.info('C value: ' + str(regr.C_[0]))

                dict_param = dict(zip(k_feature, regr.coef_[0]))
                dict_param['intercept'] = regr.intercept_

                d_view = [(v, k) for k, v in dict_param.iteritems()]
                d_view.sort(reverse=True)  #

                logging.info("")
                logging.info("Model Parameters:")
                # sorted(((v, k) for k, v in dict_param.iteritems()), reverse=True)
                for v, k in d_view:
                    if v != 0:
                        logging.info("%s: %f" % (k, v))
                logging.info("")

                if self.split_bool:
                    test_score_check.append(test_score)
                train_score_check.append(train_score)

                if not self.split_bool:
                    self.create_roc_cv_plt(X_train, y_train, regr.C_[0])

                if self.split_bool:
                    if y_feature == 'openness_group':
                        openness_score.append(test_score)
                        openness_score_roc.append(roc_score)
                    if y_feature == 'conscientiousness_group':
                        conscientiousness_score.append(test_score)
                        conscientiousness_score_roc.append(roc_score)
                    if y_feature == 'extraversion_group':
                        extraversion_score.append(test_score)
                        extraversion_score_roc.append(roc_score)
                    if y_feature == 'agreeableness_group':
                        agreeableness_score.append(test_score)
                        agreeableness_score_roc.append(roc_score)
                    if y_feature == 'neuroticism_group':
                        neuroticism_score.append(test_score)
                        neuroticism_score_roc.append(roc_score)

                if y_feature == 'openness_group':
                    openness_score_cv.append(train_score)
                if y_feature == 'conscientiousness_group':
                    conscientiousness_score_cv.append(train_score)
                if y_feature == 'extraversion_group':
                    extraversion_score_cv.append(train_score)
                if y_feature == 'agreeableness_group':
                    agreeableness_score_cv.append(train_score)
                if y_feature == 'neuroticism_group':
                    neuroticism_score_cv.append(train_score)

                '''for idx, pair_prob in enumerate(regr.predict_proba(X_test)):
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
                        false_list.append(pair_prob[pred_label])'''
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

        # from collections import Counter

        # logging.info('Counter C: ' + str(Counter(c_check)))

        '''print(true_list)
        print(len(true_list))
        print(false_list)
        print(len(false_list))'''

        # print('total ratio: ' + str(float(len(true_list))/float(len(true_list)+len(false_list))))
        return

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
                logging.info('')
                logging.info('build linear regression model for: ' + str(y_feature))

                self.raw_df = self.merge_df
                self.raw_df.to_csv(self.dir_analyze_name + 'linear_regression_final_data.csv')
                logging.info('')
                logging.info('save file: ')

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
                    logging.info('train: class 0 ratio: ' + str(sum(y_train) / len(y_train)))
                    logging.info('test: class 0 ratio: ' + str(sum(y_test) / len(y_test)))
                else:
                    X_train = X
                    y_train = y

                logging.info('all: class 0 ratio:  ' + str(sum(y)/len(y)))

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
                logging.info('')
                logging.info('Total sample size: ' + str(self.raw_df.shape[0]))
                logging.info('Number of features before selecting: ' + str(self.raw_df.shape[1]))
                logging.info('Number of k best features: ' + str(X_train.shape[1]))

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

                    logging.info('')
                    logging.info('MAE train: ' + str(mae_train))
                    logging.info('MAE test: ' + str(mae_test))
                    logging.info('MAE threshold: ' + str(mae_threshold))

                    logging.info('Pearson train: ' + str(round(pearson_c_train, 2)) + ', p val: ' + str(round(p_value_train, 3)))
                    logging.info('Pearson test: ' + str(round(pearson_c_test, 2)) + ', p val: ' + str(round(p_value_test, 3)))
                    logging.info('')

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

                logging.info("")
                logging.info("Model Parameters:")
                # sorted(((v, k) for k, v in dict_param.iteritems()), reverse=True)
                for v, k in d_view:
                    if v != 0:
                        logging.info("%s: %f" % (k, v))
                logging.info("")

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

        # logging.info('Counter C: ' + str(Counter(c_check)))

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
        logging.info('')
        logging.info('add High/Low traits group')
        logging.info('Save file: self.merge_df - ' + str(self.dir_analyze_name) + 'logistic_regression_df.csv')
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
            # logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self._cal_participant_traits_percentile_values(idx, row_participant)

        # self.merge_df = self.participant_df.copy()

        return

    # after delete un valid participant
    def cal_all_participant_percentile_value(self):
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
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

    calculate_obj.init_debug_log()                      # init log file
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
    calculate_obj.extract_item_aspect()                   # TODO add
    # calculate_obj.cal_all_participant_percentile_value()  # TODO need to add

    calculate_obj.calculate_logistic_regression()       # predict traits H or L
    # calculate_obj.calculate_linear_regression()         # predict traits using other feature
    # calculate_obj.calculate_pearson_correlation()       # calculate pearson correlation


if __name__ == '__main__':

    raise('not in use - please run using Wrapper_build_feature_dataset')
    '''
    # input file name
    participant_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_139_participant.csv'
    item_aspects_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_item_aspects.csv'
    purchase_history_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_purchase_history.csv'
    valid_users_file = '/Users/sguyelad/PycharmProjects/research/analyze_data/personality_valid_users.csv'
    dir_analyze_name = '/Users/sguyelad/PycharmProjects/research/BFI_results/analyze_pic/'
    threshold_purchase = 30
    c_value = 2
    regularization = 'l1'  # 'l2'
    bool_slice_gap_percentile = True
    bool_normalize_features = True

    main(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
         threshold_purchase, bool_slice_gap_percentile, bool_normalize_features, c_value, regularization)
    '''
