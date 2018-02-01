import sys
import csv
import logging
import pandas as pd


# create vocabularies by grouping factor
# aggregate all text together
class CreateVocabularies:

    def __init__(self, description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method,
                 verbose_flag, participants_survey_data=None, participants_ebay_mapping_file=None,
                 participants_purchase_history=None, personality_trait=None):

        self.description_file = description_file    # description file
        self.log_dir = log_dir                      # log directory
        self.directory_output = directory_output    # save description texts
        self.split_method = split_method            # split texts into groups using this method
        self.vocabulary_method = vocabulary_method  # output vocabulary separate by item description/ merge all
        self.verbose_flag = verbose_flag            # print results in addition to log file
        self.gap_value = gap_value                  # gap between low and high group - e.g 0.5 keep above .75 under .25

        self.participants_survey_data = participants_survey_data                # participants survey data
        self.participants_ebay_mapping_file = participants_ebay_mapping_file    # ueBay user name + user_id
        self.participants_purchase_history = participants_purchase_history      # history purchase
        self.personality_trait = personality_trait  # personality traits to split text by

        self.description_df = pd.DataFrame()        # contain all descriptions
        self.vertical_item_id_df = pd.DataFrame()   # load item id vertical connection

        # traits split method
        self.user_trait_df = pd.DataFrame()         # user's and their personality traits percentile
        self.user_ebay_df = pd.DataFrame()          # map eBay user name and his user_id
        self.user_item_id_df = pd.DataFrame()       # user and his items he bought

        self.item_description_dict = dict()         # dictionary contain id and html code
        self.item_text_dict = dict()                # dictionary contain id and text extracted from html code

        self.cur_time = str
        self.dir_vocabulary_name = str
        csv.field_size_limit(sys.maxsize)

    # build log object
    def init_debug_log(self):
        import logging
        from time import gmtime, strftime

        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

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

        if self.split_method not in ['vertical', 'traits']:
            raise('split method: ' + str(self.split_method) + ' is not defined')

        if self.vocabulary_method not in ['documents', 'aggregation']:
            raise('vocabulary method: ' + str(self.vocabulary_method) + ' is not defined')

        if self.gap_value > 1 or self.gap_value < 0:
            raise('gap value must be between zero to one')

        import os
        if not os.path.exists(self.directory_output):
            os.makedirs(self.directory_output)

        self.dir_vocabulary_name = self.directory_output + str(self.cur_time) + '/'
        if not os.path.exists(self.dir_vocabulary_name):
            os.makedirs(self.dir_vocabulary_name)

    # load description file
    def load_descriptions(self):

        self.description_df = pd.read_csv(self.description_file)        # items and their descriptions
        self.description_df = self.description_df[['item_id', 'description']]

        if self.split_method == 'vertical':
            self.vertical_item_id_df = pd.read_csv(self.participants_purchase_history)
            self.vertical_item_id_df = self.vertical_item_id_df[['item_id', 'BSNS_VRTCL_NAME']]

        elif self.split_method == 'traits':
            self.user_trait_df = pd.read_csv(participants_survey_data)  # participants survey data
            self.user_trait_df = self.user_trait_df[['eBay site user name', self.personality_trait + '_percentile']]

            self.user_ebay_df = pd.read_csv(participants_ebay_mapping_file)  # ueBay user name + user_id
            self.user_ebay_df = self.user_ebay_df[['USER_ID', 'USER_SLCTD_ID']]

            self.user_item_id_df = pd.read_csv(participants_purchase_history)  # history purchase
            self.user_item_id_df = self.user_item_id_df[['item_id', 'buyer_id']]

        return

    # create vocabularies using input method
    def create_vocabulary_by_method(self):
        if self.split_method == 'vertical':
            self.vertical_split_item_into_groups()
        elif self.split_method == 'traits':
            self.traits_split_item_into_groups()
        return

    # split items into groups by traits
    def traits_split_item_into_groups(self):

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

        if self.vocabulary_method == 'documents':
            cur_vocabulary_low_list = list()
            cur_vocabulary_high_list = list()
            for index, row in self.description_df.iterrows():
                if row['item_id'] in item_id_high_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_high_list.append(row['description'])

                elif row['item_id'] in item_id_low_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_low_list.append(row['description'])

            logging.info('Number of Valid description in high group: ' + str(len(cur_vocabulary_high_list)))
            logging.info('Number of Valid description in low group: ' + str(len(cur_vocabulary_low_list)))
            raise

            self.save_vocabulary('high_' + str(self.personality_trait), cur_vocabulary_high_list)
            self.save_vocabulary('low_' + str(self.personality_trait), cur_vocabulary_low_list)

        elif self.vocabulary_method == 'aggregation':
            cur_vocabulary_high = ''
            cur_vocabulary_low = ''
            for index, row in self.description_df.iterrows():
                if row['item_id'] in item_id_high_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_high += row['description']
                        cur_vocabulary_high += ' '
                elif row['item_id'] in item_id_low_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_low += row['description']
                        cur_vocabulary_low += ' '

            self.save_vocabulary('high_' + str(self.personality_trait), cur_vocabulary_high)
            self.save_vocabulary('low_' + str(self.personality_trait), cur_vocabulary_low)
        return

    # split items into groups by vertical
    def vertical_split_item_into_groups(self):

        item_description_list = list(self.description_df['item_id'])
        price_group = self.vertical_item_id_df.groupby(['BSNS_VRTCL_NAME'])

        # iterate over each vertical group
        for vertical_name, group_df in price_group:
            self.create_vocabulary(vertical_name, group_df)
        return

    # find all relevant text per group
    def create_vocabulary(self, vertical_name, group_df):

        logging.info("Vartical name: " + str(vertical_name))
        item_id_list = group_df['item_id'].tolist()  # item id's of vertical
        logging.info("Vartical size: " + str(group_df.shape[0]))

        # self.description_df['item_id'].astype(int)
        # get product description we have in the current vertical
        found_df = self.description_df.loc[self.description_df['item_id'].isin(item_id_list)]

        logging.info("Description found: " + str(found_df.shape[0]))
        logging.info("Total Description: " + str(group_df.shape[0]))
        logging.info("Fraction Description: " + str(float(found_df.shape[0]) / float(group_df.shape[0])))

        # aggregate all description per group
        if self.vocabulary_method == 'documents':
            cur_vocabulary_list = list()
            for index, row in found_df.iterrows():
                if isinstance(row['description'], basestring):
                    cur_vocabulary_list.append(row['description'])
            logging.info("Total descriptions: " + str(len(cur_vocabulary_list)))
            self.save_vocabulary(str(vertical_name), cur_vocabulary_list)

        elif self.vocabulary_method == 'aggregation':
            cur_vocabulary = ''                     # all words per group (vertical/traits)
            for index, row in found_df.iterrows():
                if isinstance(row['description'], basestring):
                    cur_vocabulary += row['description']
                    cur_vocabulary += ' '

            logging.info("Total words: " + str(len(cur_vocabulary)))
            self.save_vocabulary(str(vertical_name), cur_vocabulary)
        return

    # save vocabulary per vertical
    def save_vocabulary(self, vertical_name, cur_vocabulary):

        group_file_name = str(self.dir_vocabulary_name) + str(self.vocabulary_method) + '_' \
                          + str(vertical_name) + '.txt'

        # text_file = open(group_file_name, "w")
        # text_file.write(cur_vocabulary)
        # text_file.close()

        import pickle

        with open(group_file_name, 'wb') as fp:
            pickle.dump(cur_vocabulary, fp)



        logging.info("Save file: " + str(group_file_name))
        logging.info("")
        return


def main(description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method, verbose_flag,
         participants_survey_data=None, participants_ebay_mapping_file=None, participants_purchase_history=None,
         personality_trait=None):

    # init class
    create_vocabularies_obj = CreateVocabularies(description_file, log_dir, directory_output, split_method, gap_value,
                                                 vocabulary_method, verbose_flag, participants_survey_data,
                                                 participants_ebay_mapping_file, participants_purchase_history, personality_trait)

    create_vocabularies_obj.init_debug_log()                    # init log file
    create_vocabularies_obj.check_input()                       # check if arguments are valid
    create_vocabularies_obj.load_descriptions()                 # load description file
    create_vocabularies_obj.create_vocabulary_by_method()       # build model in regard to vertical/trait
    # create_vocabularies_obj.vertical_split_item_into_groups()   # split items into groups by vertical


if __name__ == '__main__':

    # item and hist description file

    # description_file = '/Users/sguyelad/PycharmProjects/research/kl/descriptions/num_items_447_2018-01-25 12-20-46.csv'
    # description_file = '/Users/sguyelad/PycharmProjects/research/kl/descriptions/num_items_1554_2018-01-29 14:20:27.csv'

    description_file = 'descriptions/num_items_1552_2018-01-30 13:15:33.csv'
    log_dir = 'log/'
    directory_output = 'vocabulary/'
    vocabulary_method = 'documents'     # 'documents', 'aggregation'
    verbose_flag = True
    split_method = 'traits'     # 'vertical', 'traits'
    gap_value = 0.5             # must be a float number between zero to one

    # needed if split_method is traits
    participants_survey_data = '../data/participant_data/participant_threshold_20_features_extraction.csv'  # users with more than 20 purchases
    participants_ebay_mapping_file = '../data/participant_data/personality_valid_users.csv'
    participants_purchase_history = '../data/participant_data/personality_purchase_history.csv'
    personality_trait = 'conscientiousness'  # 'agreeableness' 'extraversion' 'openness', 'conscientiousness', 'neuroticism'

    main(description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method, verbose_flag,
         participants_survey_data, participants_ebay_mapping_file, participants_purchase_history, personality_trait)
