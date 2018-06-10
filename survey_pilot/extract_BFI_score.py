from __future__ import print_function
import pandas as pd
import logging
import numpy as np


class CalculateBFIScore:

    """
    Calculate BFI score and percentile

    Args:
        participant_file: csv file contain user and his BFI test results (user per row)
        dir_save_results: directory path to save

    Returns:
        csv: contain original data and traits value and percentile values, path of dir_save_results

    Raises:
    """

    def __init__(self, participant_file, dir_save_results, verbose_flag=True):

        # file arguments
        self.participant_file = participant_file
        self.dir_save_results = dir_save_results
        self.verbose_flag = verbose_flag

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

        self.cur_time = None

    # build log object
    def init_debug_log(self):
        import logging

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        log_dir = 'log/'
        log_file_name = log_dir + 'extract_BFI_score_' + str(self.cur_time) + '.log'

        logging.basicConfig(filename=log_file_name,
                            filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

        # print result in addition to log file
        if self.verbose_flag:
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

        logging.info("")
        logging.info("")
        logging.info("start log program")

    # load csv into df
    def load_clean_csv_results(self):
        self.participant_df = pd.read_csv(self.participant_file)

    # extract valid user_name only - currently not in use
    def clean_df(self):

        return
        # extract only valid user name
        for (idx, row_participant) in self.participant_df.iterrows():
            lower_first_name = row_participant['eBay site user name'].lower()
            self.participant_df.at(idx, 'eBay site user name', lower_first_name)

        self.participant_df = self.participant_df[self.participant_df['eBay site user name'].isin(self.valid_user_list)]

    # reverse all relevant question values
    def change_reverse_value(self):
        reverse_col = [2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43]
        for cur_rcol in reverse_col:
            start_str_cur = str(cur_rcol) + '.'
            filter_col = [col for col in self.participant_df if col.startswith(start_str_cur)][0]
            logging.info('Change column values (reverse mode): ' + str(filter_col))
            self.participant_df[filter_col] = self.participant_df[filter_col].apply(lambda x: 6 - x)

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

        # add average traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate traits value for participant: ' + str(row_participant['Email address']))
            self._calculate_individual_score(idx, row_participant)

        # add percentile traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self._cal_participant_traits_percentile_values(idx, row_participant)

        self.merge_df = self.participant_df.copy()

        import os
        dir_path = self.dir_save_results + '/participant_bfi_score/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        output_file = dir_path + 'users_with_bfi_score_amount_' + str(self.merge_df.shape[0]) + '.csv'
        self.merge_df.to_csv(output_file)
        logging.info('save BFI score into file: ' + str(output_file))

    # calculate traits values for one participant
    def _calculate_individual_score(self, idx, row_participant):

        op_trait = self.cal_participant_traits(row_participant, self.question_openness, self.ratio_hundred_openness)

        self.participant_df.at[idx, 'openness_trait'] = op_trait
        self.openness_score_list.append(op_trait)

        co_trait = self.cal_participant_traits(row_participant, self.question_conscientiousness,
                                               self.ratio_hundred_conscientiousness)
        self.participant_df.at[idx, 'conscientiousness_trait'] = co_trait
        self.conscientiousness_score_list.append(co_trait)

        ex_trait = self.cal_participant_traits(row_participant, self.question_extraversion,
                                               self.ratio_hundred_extraversion)
        self.participant_df.at[idx, 'extraversion_trait'] = ex_trait
        self.extraversion_score_list.append(ex_trait)

        ag_trait = self.cal_participant_traits(row_participant, self.question_agreeableness,
                                               self.ratio_hundred_agreeableness)
        self.participant_df.at[idx, 'agreeableness_trait'] = ag_trait
        self.agreeableness_score_list.append(ag_trait)

        ne_trait = self.cal_participant_traits(row_participant, self.question_neuroticism,
                                               self.ratio_hundred_neuroticism)

        self.participant_df.at[idx, 'neuroticism_trait'] = ne_trait
        self.neuroticism_score_list.append(ne_trait)

    # after delete un valid participant
    def _cal_all_participant_percentile_value(self):
        for (idx, row_participant) in self.participant_df.iterrows():
            logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self._cal_participant_traits_percentile_values(idx, row_participant)

    # calculate percentile value for one participant
    def _cal_participant_traits_percentile_values(self, idx, participant_score):

        op_per = float(sum(i < participant_score['openness_trait'] for i in self.openness_score_list))/float(len(self.openness_score_list)-1)
        self.participant_df.at[idx, 'openness_percentile'] = op_per
        co_per = float(sum(
            i < participant_score['conscientiousness_trait'] for i in self.conscientiousness_score_list))/float(len(self.conscientiousness_score_list)-1)
        self.participant_df.at[idx, 'conscientiousness_percentile'] = co_per
        ex_per = float(sum(
            i < participant_score['extraversion_trait'] for i in self.extraversion_score_list))/float(len(self.extraversion_score_list)-1)
        self.participant_df.at[idx, 'extraversion_percentile'] = ex_per
        ag_per = float(sum(
            i < participant_score['agreeableness_trait'] for i in self.agreeableness_score_list))/float(len(self.agreeableness_score_list)-1)
        self.participant_df.at[idx, 'agreeableness_percentile'] = ag_per
        ne_per = float(sum(
            i < participant_score['neuroticism_trait'] for i in self.neuroticism_score_list))/float(len(self.neuroticism_score_list)-1)
        self.participant_df.at[idx, 'neuroticism_percentile'] = ne_per

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
        """
        check duplication and the difference in results
        TODO delete users and save only relevant users results
        :return:
        """
        count_dup = 0
        dup_group = self.participant_df.groupby(['Email address'])
        # dup_group = self.participant_df.groupby(['eBay site user name'])#['Email address'])
        # iterate over each user
        for email_name, group in dup_group:
            if group.shape[0] > 1:
                count_dup += 1
                logging.info(email_name)
                for (idx, row_participant) in group.iterrows():
                    logging.info(row_participant['eBay site user name'])
                    logging.info(list(row_participant[['openness_trait', 'conscientiousness_trait', 'extraversion_trait',
                                                'agreeableness_trait', 'neuroticism_trait', 'openness_percentile',
                                                'conscientiousness_percentile',	'extraversion_percentile',
                                                'agreeableness_percentile',	'neuroticism_percentile']]))

                logging.info('')

        logging.info('Number of duplication: ' + str(count_dup))


def main(participant_file, dir_save_results):

    calculate_obj = CalculateBFIScore(participant_file, dir_save_results)    # create object and variables

    calculate_obj.init_debug_log()                      # init log file
    calculate_obj.load_clean_csv_results()              # load data set
    calculate_obj.clean_df()                            # clean df - e.g. remain valid users only
    calculate_obj.change_reverse_value()                # change specific column into reverse mode
    calculate_obj.cal_participant_traits_values()       # calculate average traits and percentile value

    # after I saved the file with percentile value
    calculate_obj.investigate_duplication()


if __name__ == '__main__':

    # input file name
    participant_file = '../data/participant_data/1425 users input/personality_participant_all_include_1287_CF total_1425.csv'
    dir_save_results = '../results/BFI_results/'

    main(participant_file, dir_save_results)