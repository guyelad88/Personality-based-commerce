import sys
import csv
import logging
import pandas as pd


# build data set
# for each trait split descriptions into two groups (with/no gap)
class BuildDataSet:

    def __init__(self, description_file, log_dir, directory_output, gap_value, verbose_flag,
         participants_survey_data, participants_ebay_mapping_file, participants_purchase_history):

        self.description_file = description_file    # description file
        self.log_dir = log_dir                      # log directory
        self.directory_output = directory_output    # save description texts
        self.verbose_flag = verbose_flag            # print results in addition to log file
        self.gap_value = gap_value                  # gap between low and high group - e.g 0.5 keep above .75 under .25

        self.traits_list = ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']  #

        self.personality_trait = str()

        self.participants_survey_data = participants_survey_data                # participants survey data
        self.participants_ebay_mapping_file = participants_ebay_mapping_file    # ueBay user name + user_id
        self.participants_purchase_history = participants_purchase_history      # history purchase - item_id + item_data (vertical, price, etc.)

        self.description_df = pd.DataFrame()        # contain all descriptions
        self.vertical_item_id_df = pd.DataFrame()   # load item id vertical connection

        # traits split method
        self.user_trait_df = pd.DataFrame()         # user's and their personality traits percentile
        self.user_ebay_df = pd.DataFrame()          # map eBay user name and his user_id
        self.user_item_id_df = pd.DataFrame()       # user and his items he bought
        self.full_user_item_id_df = pd.DataFrame()  # item with all purchase data

        self.item_description_dict = dict()         # dictionary contain id and html code
        self.item_text_dict = dict()                # dictionary contain id and text extracted from html code

        self.users_traits = dict()                  # users and their amount per trait

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.dir_vocabulary_name = str
        csv.field_size_limit(sys.maxsize)

    # build log object
    def init_debug_log(self):
        import logging

        lod_file_name = self.log_dir + 'create_vocabularies_' + str(self.cur_time) + '.log'

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

    def check_input(self):

        if self.gap_value > 1 or self.gap_value < 0:
            raise('gap value must be between zero to one')

        import os
        if not os.path.exists(self.directory_output):
            os.makedirs(self.directory_output)

        self.dir_vocabulary_name = self.directory_output + str(self.cur_time) + '/'
        if not os.path.exists(self.dir_vocabulary_name):
            os.makedirs(self.dir_vocabulary_name)

        if not os.path.exists(self.dir_vocabulary_name + 'histogram/'):
            os.makedirs(self.dir_vocabulary_name + 'histogram/')

        return

    # load description file
    def load_descriptions(self):

        self.description_df = pd.read_csv(self.description_file)                # items and their descriptions
        self.description_df = self.description_df[['item_id', 'description']]

        self.user_trait_df = pd.read_csv(self.participants_survey_data)         # participants survey data
        # self.user_trait_df = self.user_trait_df[['eBay site user name', self.personality_trait + '_percentile']]

        self.user_ebay_df = pd.read_csv(self.participants_ebay_mapping_file)    # ueBay user name + user_id
        self.user_ebay_df = self.user_ebay_df[['USER_ID', 'USER_SLCTD_ID']]

        self.user_item_id_df = pd.read_csv(self.participants_purchase_history)  # history purchase
        self.full_user_item_id_df = self.user_item_id_df.copy(deep=True)
        self.user_item_id_df = self.user_item_id_df[['item_id', 'buyer_id']]

        return

    def create_data_set_per_trait(self):

        for idx, traits in enumerate(self.traits_list):

            logging.info('Create data set for personality trait: ' + str(traits))
            self.personality_trait = traits

            self.users_traits[traits] = dict()          #
            self.users_traits[traits]['high'] = dict()  #
            self.users_traits[traits]['low'] = dict()   #

            high_limit = 0.5 + (self.gap_value / 2)
            low_limit = 0.5 - (self.gap_value / 2)

            logging.info('High limit: ' + str(high_limit) + ', ' + 'Low limit: ' + str(low_limit))
            # user name list per group - traits values
            user_name_high_percentile_list = \
                self.user_trait_df.loc[self.user_trait_df[self.personality_trait + '_percentile'] > high_limit][
                    'eBay site user name'].tolist()
            user_name_low_percentile_list = \
                self.user_trait_df.loc[self.user_trait_df[self.personality_trait + '_percentile'] < low_limit][
                    'eBay site user name'].tolist()

            # user id per group
            user_id_high_percentile_list = list()
            user_id_low_percentile_list = list()

            for index, user_row in self.user_ebay_df.iterrows():
                if user_row['USER_SLCTD_ID'] in user_name_high_percentile_list:
                    user_id_high_percentile_list.append(user_row['USER_ID'])
                elif user_row['USER_SLCTD_ID'] in user_name_low_percentile_list:
                    user_id_low_percentile_list.append(user_row['USER_ID'])

            # item id per group
            item_id_high_percentile_list = list()
            item_id_low_percentile_list = list()

            # get items per group
            for index, user_row in self.user_item_id_df.iterrows():
                if user_row['buyer_id'] in user_id_high_percentile_list:
                    item_id_high_percentile_list.append(user_row['item_id'])
                elif user_row['buyer_id'] in user_id_low_percentile_list:
                    item_id_low_percentile_list.append(user_row['item_id'])

            logging.info('Potential Number of description in high group: ' + str(len(item_id_high_percentile_list)))
            logging.info('Potential Number of description in low group: ' + str(len(item_id_low_percentile_list)))

            cur_vocabulary_low_list = list()
            cur_vocabulary_high_list = list()

            for index, row in self.description_df.iterrows():
                if row['item_id'] in item_id_high_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_high_list.append(row['description'])
                        self.update_users_in_groups(row['item_id'], 'high', user_id_high_percentile_list)

                elif row['item_id'] in item_id_low_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_low_list.append(row['description'])
                        self.update_users_in_groups(row['item_id'], 'low', user_id_low_percentile_list)

            logging.info('Number of Valid description in high group: ' + str(len(cur_vocabulary_high_list)))
            logging.info('Number of Valid description in low group: ' + str(len(cur_vocabulary_low_list)))

            # create dir for both elements
            import os
            relevant_dir = str(self.dir_vocabulary_name) + str(self.personality_trait) + '/'
            if not os.path.exists(relevant_dir):
                os.makedirs(relevant_dir)

            self.save_vocabulary('high_' + str(self.personality_trait), cur_vocabulary_high_list, relevant_dir, len(cur_vocabulary_high_list))
            self.save_vocabulary('low_' + str(self.personality_trait), cur_vocabulary_low_list, relevant_dir, len(cur_vocabulary_low_list))

            self.save_user_trait_statistics('high')
            self.save_user_trait_statistics('low')

        return

    # count user amount in groups
    def update_users_in_groups(self, item_id, group_type, user_group_list):

        cur_row = self.user_item_id_df.loc[self.user_item_id_df['item_id'] == item_id]
        already_insert = list()

        for index, user_row in cur_row.iterrows():

            if user_row['buyer_id'] in already_insert:          # relevant for two products TODO think about it
                continue

            if user_row['buyer_id'] in user_group_list:
                already_insert.append(user_row['buyer_id'])     # avoid duplication TODO check if needed

                if user_row['buyer_id'] not in self.users_traits[self.personality_trait][group_type]:
                    self.users_traits[self.personality_trait][group_type][user_row['buyer_id']] = 1
                else:
                    self.users_traits[self.personality_trait][group_type][user_row['buyer_id']] += 1

        return

    def save_user_trait_statistics(self, group_type):

        import matplotlib.pyplot as plt
        plt.figure()
        histogram_data = self.users_traits[self.personality_trait][group_type].values()
        unique_users = len(self.users_traits[self.personality_trait][group_type].keys())
        sum_items = sum(self.users_traits[self.personality_trait][group_type].values())
        plt.hist(histogram_data, bins=30)
        plt.ylabel('Amount')
        plt.xlabel('Number of purchase')
        plt.title(str(self.personality_trait) + ' ' + str(group_type) +
                  ': Histogram number of purchases, Unique Users: ' +
                  str(unique_users))
        # plt.show()

        group_file_name = str(self.dir_vocabulary_name) + 'histogram/' + str(self.personality_trait) + '_' + str(group_type) + \
                          '_unique_users_' + str(unique_users) + '_number_items_' + str(sum_items) + '_gap_'\
                          + str(self.gap_value) + '.png'

        plt.savefig(group_file_name)
        logging.info("Save histogram: " + str(group_file_name))
        logging.info("")

        return

    # save vocabulary per vertical
    def save_vocabulary(self, vertical_name, cur_vocabulary, relevant_dir, number_desc):

        group_file_name = str(relevant_dir) + str(vertical_name) + '_gap_' + str(self.gap_value) + '_' + str(number_desc) + '.txt'

        import pickle

        with open(group_file_name, 'wb') as fp:
            pickle.dump(cur_vocabulary, fp)

        logging.info("Save file: " + str(group_file_name))
        logging.info("")

        return


def main(description_file, log_dir, directory_output, gap_value, verbose_flag,
         participants_survey_data=None, participants_ebay_mapping_file=None, participants_purchase_history=None):

    # init class
    build_data_set_obj = BuildDataSet(description_file, log_dir, directory_output, gap_value, verbose_flag,
         participants_survey_data, participants_ebay_mapping_file, participants_purchase_history)

    build_data_set_obj.init_debug_log()                     # init log file
    build_data_set_obj.check_input()                        # check if arguments are valid
    build_data_set_obj.load_descriptions()                  # load description file
    build_data_set_obj.create_data_set_per_trait()          # create data set for each trait seperatly

if __name__ == '__main__':

    # item and hist description file

    description_file = 'descriptions/num_items_1552_2018-01-30 13:15:33.csv'
    log_dir = 'log/'
    directory_output = 'dataset/'
    verbose_flag = True
    gap_value = 0.5             # must be a float number between zero to one

    # needed if split_method is traits
    participants_survey_data = '../data/participant_data/participant_threshold_20_features_extraction.csv'  # users with more than 20 purchases
    participants_ebay_mapping_file = '../data/participant_data/personality_valid_users.csv'
    participants_purchase_history = '../data/participant_data/personality_purchase_history.csv'

    main(description_file, log_dir, directory_output, gap_value, verbose_flag,
         participants_survey_data, participants_ebay_mapping_file, participants_purchase_history)
