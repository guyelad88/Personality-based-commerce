from __future__ import print_function
import pandas as pd
import os
import numpy as np
import lexrank_config
from time import gmtime, strftime
from utils.logger import Logger


class CalculateBFIScore:

    """
    Calculate BFI score and percentile for each user
    Compute duplications - analyze whether valid or not and remove users accordingly
    before merge data sets (determine the real users and choose for users with duplication their real score)

    Args:
        participant_file: csv file contain user and his BFI test results (user per row)
        dir_save_results: directory path to save
        threshold: max gap too differ between good and bad users
        name_length_threshold: exclude name too short

    Returns:
        1. contain original data + traits value and percentile values
            ../results/data/BFI_results/participant_bfi_score

        2. remain only good users (after filtering duplication and bad user names)
            ../results/data/BFI_results/participant_bfi_score_clean_duplication
    """

    def __init__(self, participant_file, dir_save_results, threshold, name_length_threshold, verbose_flag=True):

        # file arguments
        self.participant_file = participant_file
        self.dir_save_results = dir_save_results
        self.verbose_flag = verbose_flag
        self.threshold = threshold
        self.name_length_threshold = name_length_threshold

        # define data frame needed for analyzing data
        self.participant_df = pd.DataFrame()
        self.item_aspects_df = pd.DataFrame()
        self.purchase_history_df = pd.DataFrame()
        self.valid_users_df = pd.DataFrame()
        self.clean_participant_data = pd.DataFrame()

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

        self.dup_gap_list = list()          # duplication with their max gap

        self.valid_user_list = list()       # valid users

        self.item_buyer_dict = dict()       # item-buyer dict 1:N
        self.user_name_id_dict = dict()     # missing data because it 1:N
        self.user_id_name_dict = dict()     #

        self.traits_list = [
            'agreeableness',
            'extraversion',
            'openness',
            'conscientiousness',
            'neuroticism'
        ]

        self.cur_time = None
        self.log_file_name = str()

    # build log object
    def init_debug_log(self):
        """ create a Logger object and save log in a file """

        file_prefix = 'extract_bfi_score'
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_file_name = '../log/{}_{}.log'.format(file_prefix, self.cur_time)
        Logger.set_handlers('ExtractBFIScore', self.log_file_name, level='debug')

    # load csv into df
    def load_clean_csv_results(self):
        self.participant_df = pd.read_csv(self.participant_file)
        self.participant_df['eBay site user name'] = self.participant_df['eBay site user name'].str.lower()

    # TODO not in use - extract valid user_name only - currently not in use
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
            Logger.info('Change column values (reverse mode): ' + str(filter_col))
            self.participant_df[filter_col] = self.participant_df[filter_col].apply(lambda x: 6 - x)

    # calculate traits valuers and percentile per participant
    def cal_participant_traits_values(self):

        # self.participant_df = self.participant_df.head(150)
        Logger.info('DF size: ' + str(self.participant_df.shape))
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

        # add traits value columns
        for (idx, row_participant) in self.participant_df.iterrows():
            if idx % 100 == 0:
                Logger.info('Calculate traits value for participant number: {} / {}'.format(
                    str(idx),
                    str(self.participant_df.shape[0])))
            self._calculate_individual_score(idx, row_participant)

        # add percentile traits columns
        for (idx, row_participant) in self.participant_df.iterrows():
            if idx % 100 == 0:
                Logger.info('Calculate percentile value for participant number: {} / {}'.format(
                    str(idx),
                    str(self.participant_df.shape[0])))

            self._cal_participant_traits_percentile_values(idx, row_participant)

        self.merge_df = self.participant_df.copy()

        dir_path = self.dir_save_results + 'participant_bfi_score/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save all users with duplication
        output_file = '{}users_with_bfi_score_amount_{}_time_{}.csv'.format(
            dir_path,
            str(self.merge_df.shape[0]),
            str(self.cur_time)
        )

        self.merge_df.to_csv(output_file)

        Logger.info('save BFI score into file: ' + str(output_file))

    # calculate traits values for one participant
    def _calculate_individual_score(self, idx, row_participant):

        op_trait = self._cal_participant_traits(row_participant, self.question_openness, self.ratio_hundred_openness)

        self.participant_df.at[idx, 'openness_trait'] = op_trait
        self.openness_score_list.append(op_trait)

        co_trait = self._cal_participant_traits(row_participant, self.question_conscientiousness,
                                               self.ratio_hundred_conscientiousness)
        self.participant_df.at[idx, 'conscientiousness_trait'] = co_trait
        self.conscientiousness_score_list.append(co_trait)

        ex_trait = self._cal_participant_traits(row_participant, self.question_extraversion,
                                               self.ratio_hundred_extraversion)
        self.participant_df.at[idx, 'extraversion_trait'] = ex_trait
        self.extraversion_score_list.append(ex_trait)

        ag_trait = self._cal_participant_traits(row_participant, self.question_agreeableness,
                                               self.ratio_hundred_agreeableness)
        self.participant_df.at[idx, 'agreeableness_trait'] = ag_trait
        self.agreeableness_score_list.append(ag_trait)

        ne_trait = self._cal_participant_traits(row_participant, self.question_neuroticism,
                                               self.ratio_hundred_neuroticism)

        self.participant_df.at[idx, 'neuroticism_trait'] = ne_trait
        self.neuroticism_score_list.append(ne_trait)

    # after delete un valid participant
    def _cal_all_participant_percentile_value(self):
        for (idx, row_participant) in self.participant_df.iterrows():
            Logger.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
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
    def _cal_participant_traits(self, row, cur_trait_list, ratio):
        trait_val = 0
        for cur_col in cur_trait_list:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.participant_df if col.startswith(start_str_cur)][0]   # find col name
            trait_val += row[filter_col]
        trait_val = float(trait_val)/float(len(cur_trait_list))     # mean of traits
        return trait_val

    # handler of duplication users - who to delete and keep
    def investigate_duplication(self):
        """
        extract valid participants (remove bad duplication and bad user name)
        save df with valid user only
        """
        self.clean_participant_data = pd.DataFrame(columns=list(self.participant_df))

        Logger.info('check duplication, max threshold: ' + str(self.threshold))
        dup_above_gap = 0               # users with dup, not OK
        dup_below_gap = 0               # uses with dup, but OK
        no_dup = 0                      # users appears only ones
        cnt_len_below_threshold = 0     # user name length is too short - heuristic

        # self.participant_df['eBay site user name'].str.lower()
        ebay_user_group = self.participant_df.groupby(['eBay site user name'])

        # iterate over each user and check max gap in duplication
        for ebay_user_name, user_group in ebay_user_group:

            if len(ebay_user_name) < self.name_length_threshold:
                Logger.info('remove because name is too short: {}'.format(str(ebay_user_name)))
                cnt_len_below_threshold += 1
                continue

            if user_group.shape[0] > 1:     # user name with duplication

                max_gap, diff_dup = self._check_max_gap(user_group)

                # check max gap - decide whether to insert into clean_df or not
                if max_gap > self.threshold:
                    Logger.info('Duplication user above threshold: {} : {}, amount: {}'.format(
                        str(ebay_user_name),
                        str(max_gap),
                        str(user_group.shape[0])))
                    dup_above_gap += 1
                    continue

                else:
                    Logger.info('Duplication user below threshold: {} : {}, amount: {}'.format(
                        str(ebay_user_name),
                        str(max_gap),
                        str(user_group.shape[0])))
                    dup_below_gap += 1

                # insert duplication users (his first appearance)
                first_in_group = user_group.head(1)
                self.clean_participant_data = self.clean_participant_data.append(first_in_group)

            else:               # user name without duplication
                no_dup += 1
                # Logger.info('Unique user: ' + str(ebay_user_name))
                first_in_group = user_group.head(1)
                self.clean_participant_data = self.clean_participant_data.append(first_in_group)

        Logger.info('')
        Logger.info('summary:')
        Logger.info('Clean samples: ' + str(no_dup + dup_below_gap))
        Logger.info('Duplication users valid: ' + str(dup_below_gap))
        Logger.info('Duplication users not valid: ' + str(dup_above_gap))
        Logger.info('Duplication users valid ratio: ' + str(float(dup_below_gap)/float(dup_below_gap+dup_above_gap)))
        Logger.info('One users valid: ' + str(no_dup))
        Logger.info('Users with user name not valid (below length threshold): ' + str(cnt_len_below_threshold))

        dir_path = self.dir_save_results + 'participant_bfi_score_clean_duplication/'

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = '{}clean_participant_{}_{}.csv'.format(
            dir_path,
            str(self.clean_participant_data.shape[0]),
            str(self.cur_time)
        )

        self.clean_participant_data.to_csv(file_name, index=False)

        Logger.info('remove users with duplication and save: ' + str(file_name))

    def _check_max_gap(self, user_group):
        """ calculate user diff between traits values for all his appearance in data set """
        # init trait dictionary values
        diff_dup = {
            'agreeableness': list(),
            'extraversion': list(),
            'openness': list(),
            'conscientiousness': list(),
            'neuroticism': list()
        }

        # store all user values
        for (idx, row_participant) in user_group.iterrows():
            for trait in self.traits_list:
                cur_f = trait + '_trait'
                diff_dup[trait].append(row_participant[cur_f])

        # compute max diff
        max_gap = 0
        for c_t, v_l in diff_dup.iteritems():
            if max_gap < max(v_l) - min(v_l):
                max_gap = max(v_l) - min(v_l)

        self.dup_gap_list.append(max_gap)   # store to create histogram of max diff

        return max_gap, diff_dup


def main(participant_file, dir_save_results, threshold, name_length_threshold):

    calculate_obj = CalculateBFIScore(participant_file, dir_save_results, threshold, name_length_threshold)

    calculate_obj.init_debug_log()                      # init log file
    calculate_obj.load_clean_csv_results()              # load data set
    calculate_obj.clean_df()                            # TODO not in use - clean df - e.g. remain valid users only
    calculate_obj.change_reverse_value()                # change specific column into reverse mode
    calculate_obj.cal_participant_traits_values()       # calculate average traits and percentile value

    # after saved the file with percentile value
    calculate_obj.investigate_duplication()


if __name__ == '__main__':

    # input file name
    # participant_file = '../data/participant_data/1425 users input/personality_participant_all_include_1287_CF total_1425.csv'

    # 7.12
    participant_file = '/Users/gelad/Personality-based-commerce/data/participant_data/29_8_1018/CF_Ebay _Technion_1665_users.csv'

    dir_save_results = '../results/data/BFI_results/'

    threshold = lexrank_config.extract_big_five_inventory_score['threshold']
    name_length_threshold = lexrank_config.extract_big_five_inventory_score['name_length_threshold']

    main(participant_file, dir_save_results, threshold, name_length_threshold)
