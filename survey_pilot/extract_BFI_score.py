from __future__ import print_function
import pandas as pd
import logging
import numpy as np


class CalculateBFIScore:

    def __init__(self, participant_file, dir_save_results):

        # file arguments
        self.participant_file = participant_file
        self.dir_save_results = dir_save_results

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
        logging.basicConfig(filename='/Users/sguyelad/PycharmProjects/Personality-based-commerce/survey_pilot/log/analyze_results.log',
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

        '''
        import matplotlib.pyplot as plt
        # d = pd.Series(data=np.random.rand(10), index=range(10))
        a = list(self.participant_df['Nationality'])

        from collections import Counter
        # a = ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
        letter_counts = Counter(a)
        df = pd.DataFrame.from_dict(letter_counts, orient='index').sort_index()
        df.plot(kind='bar')
        plt.xticks(rotation=70)
        plt.show()
        raise'''

        return

    def clean_df(self):

        return

        '''
        # use only valid user id
        tmp_valid_user_list = list(self.valid_users_df['USER_SLCTD_ID'])
        self.valid_user_list = [x for x in tmp_valid_user_list if str(x) != 'nan']
        '''

        # extract only valid user name
        for (idx, row_participant) in self.participant_df.iterrows():
            lower_first_name = row_participant['eBay site user name'].lower()
            self.participant_df.set_value(idx, 'eBay site user name', lower_first_name)

        self.participant_df = self.participant_df[self.participant_df['eBay site user name'].isin(self.valid_user_list)]

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

        # self.merge_df.to_csv('/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/' +
        #                     'merge_df_crowdflower_' + str(self.merge_df.shape[0]) + '.csv')
        output_file = dir_save_results + 'merge_df_crowdflower_' + str(self.merge_df.shape[0]) + '.csv'
        self.merge_df.to_csv(output_file)
        logging.info('save BFI score into file: ' + str(output_file))
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

    def investigate_duplication(self):
        '''
        check duplication and the difference in results
        :return:
        '''
        count_dup = 0
        dup_group = self.participant_df.groupby(['Email address'])
        # dup_group = self.participant_df.groupby(['eBay site user name'])#['Email address'])
        # iterate over each user
        for email_name, group in dup_group:
            if group.shape[0] > 1:
                count_dup += 1
                print(email_name)
                for (idx, row_participant) in group.iterrows():
                    print(row_participant['eBay site user name'])
                    print(list(row_participant[['openness_trait', 'conscientiousness_trait', 'extraversion_trait',
                                                'agreeableness_trait', 'neuroticism_trait', 'openness_percentile',
                                                'conscientiousness_percentile',	'extraversion_percentile',
                                                'agreeableness_percentile',	'neuroticism_percentile']]))

                print('')

        print('Number of duplication: ' + str(count_dup))
        return


def main(participant_file, dir_save_results):

    calculate_obj = CalculateBFIScore(participant_file, dir_save_results)    # create object and variables

    calculate_obj.init_debug_log()                      # init log file
    calculate_obj.load_clean_csv_results()              # load data set
    calculate_obj.clean_df()                            # clean df - e.g. remain valid users only
    calculate_obj.change_reverse_value()                # change specific column into reverse mode
    calculate_obj.cal_participant_traits_values()       # calculate average traits and percentile value

    calculate_obj.investigate_duplication()             #

if __name__ == '__main__':

    # input file name
    # participant_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/crowdflower data/Personality test (BFI) - Crowdflower 113 participant.csv'
    # participant_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/merge_df_crowdflower_145.csv'
    participant_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/1425 users input/personality_participant_all_include_1287_CF total_1425.csv'
    dir_save_results = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/1425 users input/'

    main(participant_file, dir_save_results)