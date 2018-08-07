import os
import sys
import csv
import pandas as pd
import pickle
import numpy as np
from time import gmtime, strftime

import config
from utils.logger import Logger

GAP_VALUE = config.create_vocabularies['gap_value']
# SPLIT_METHOD = config.create_vocabularies['split_method']
# PERSONALITY_TRAIT = config.create_vocabularies['personality_trait']
# VERTICAL = config.create_vocabularies['vertical']
# VOCABULARY_METHOD = config.create_vocabularies['vocabulary_method']

PERSONALITY_TRAIT_LIST = config.personality_trait       # list of the big five PT


# create vocabularies by grouping factor
# aggregate all text together
class CreateVocabularies:
    """
    Create descriptions vocabulary regards to: vertical / traits / vertical + traits
    Save the two groups in a csv files using pickle package

    Args:
        merge_df: csv with clean description (extract from HTML)
        log_dir: where to save log file
        directory_output: dir to save groups excels
        split_method: split descriptions using trait/vertical/both
        gap_value: gap determine to choose users percentile threshold  - e.g. 0.5 --> val<0.25 or val>0.75
        vocabulary_method: output vocabulary separate by item description ('description') / merge all ('aggregate')
        verbose_flag: whether print log to console

        # default arguments (optional)

        participants_survey_data=None: TODO
        participants_ebay_mapping_file=None: TODO
        participants_purchase_history=None: TODO

        personality_trait=None: which traits to split by
        vertical=None: which vertical to split by
        cur_time=None: time (embed to output file/log) - easy to recognize later

    Returns:
        csv: contain original data and traits value and percentile values, path of dir_save_results
             results/vocabulary/trait_name/____.txt

    Raises:
        gap value isn't float between 0-1.
        vertical | trait isn't valud name
        split methods not defined
    """

    def __init__(self, merge_df_path):

        self.merge_df_path = merge_df_path    # description file
        self.directory_output = '../results/vocabulary/'    # save description texts

        self.split_method = SPLIT_METHOD            # split texts into groups using this method
        self.vocabulary_method = VOCABULARY_METHOD  # output vocabulary separate by item description/ merge all
        self.gap_value = GAP_VALUE                  # gap between low and high group - e.g 0.5 keep above .75 under .25
        self.personality_trait = PERSONALITY_TRAIT  # personality traits to split text by
        self.vertical = VERTICAL  # vertical to split by

        self.merge_df = pd.DataFrame()

        self.item_description_dict = dict()         # dictionary contain id and html code
        self.item_text_dict = dict()                # dictionary contain id and text extracted from html code

        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.dir_vocabulary_name = str()
        self.log_file_name = str()
        csv.field_size_limit(sys.maxsize)

    # build log object
    def init_debug_log(self):
        """ create a Logger object and save log in a file """
        file_prefix = 'create_vocabularies'
        self.log_file_name = '../log/{}_{}.log'.format(file_prefix, self.cur_time)
        Logger.set_handlers('CreateVocabularies', self.log_file_name, level='debug')

    def check_input(self):

        if self.split_method not in ['vertical', 'traits', 'traits_vertical']:
            raise ValueError('split method: {} is not defined'.format(str(self.split_method)))

        if self.vocabulary_method not in ['documents', 'aggregation']:
            raise ValueError('vocabulary method: {} is not defined'.format(str(self.vocabulary_method)))

        if self.gap_value > 1 or self.gap_value < 0:
            raise ValueError('gap value must be between zero to one')

        if not os.path.exists(self.directory_output):
            os.makedirs(self.directory_output)

        self.dir_vocabulary_name = '{}{}/'.format(self.directory_output, str(self.personality_trait))
        if not os.path.exists(self.dir_vocabulary_name):
            os.makedirs(self.dir_vocabulary_name)

    # load description file
    def load_descriptions(self):

        self.merge_df = pd.read_csv(self.merge_df_path)        # items and their descriptions

        """if self.split_method == 'vertical':
            self.vertical_item_id_df = pd.read_csv(self.participants_purchase_history)
            self.vertical_item_id_df = self.vertical_item_id_df[['item_id', 'BSNS_VRTCL_NAME']]

        elif self.split_method in ['traits', 'traits_vertical']:
            self.user_trait_df = pd.read_csv(self.participants_survey_data)  # participants survey data
            self.user_trait_df = self.user_trait_df[['eBay site user name', self.personality_trait + '_percentile']]

            self.user_ebay_df = pd.read_csv(self.participants_ebay_mapping_file)  # ueBay user name + user_id
            self.user_ebay_df = self.user_ebay_df[['USER_ID', 'USER_SLCTD_ID']]

            self.user_item_id_df = pd.read_csv(self.participants_purchase_history)  # history purchase
            self.full_user_item_id_df = self.user_item_id_df.copy(deep=True)
            self.user_item_id_df = self.user_item_id_df[['item_id', 'buyer_id']]

        self.user_item_id_df['item_id'] = self.user_item_id_df['item_id'].astype(int)
        self.full_user_item_id_df['item_id'] = self.full_user_item_id_df['item_id'].astype(int)
        """
        return

    # main function - create vocabularies using input method
    def create_vocabulary_by_method(self):
        if self.split_method == 'traits':
            self.traits_split_item_into_groups()
        else:
            raise ValueError('currently support only on traits')
        """
        elif self.split_method == 'vertical':
            self.vertical_split_item_into_groups()
        elif self.split_method == 'traits_vertical':
            self.traits_vertical_split_item_into_groups()
        """
        return

    @staticmethod
    def _create_folder_and_save(df, dir_path, file_name, log_title):
        """ save output data  """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_full_path = '{}{}'.format(dir_path, file_name)
        df.to_csv(file_full_path, index=False)
        Logger.info('{}: {}'.format(log_title, str(file_name)))

        return file_full_path

    # split items into groups by traits
    @staticmethod
    def traits_split_item_into_groups(merge_df_path, log_file_name, level='info'):
        """
        1. add a column to each trait - 'H'/'L'/'M'
        """

        Logger.set_handlers('AddPersonalityTraitGroups', log_file_name, level=level)

        merge_df = pd.read_csv(merge_df_path)

        high_limit = 0.5 + (GAP_VALUE / 2)
        low_limit = 0.5 - (GAP_VALUE / 2)

        Logger.info('High limit: {}, Low limit: {}'.format(str(high_limit), str(low_limit)))

        for p_t in PERSONALITY_TRAIT_LIST:
            source_col_name = '{}_percentile'.format(p_t)
            target_col_name = '{}_group'.format(p_t)

            merge_df[target_col_name] = ''
            merge_df[target_col_name] = np.where(merge_df[source_col_name] < low_limit, 'L',
                                       np.where(merge_df[source_col_name] < high_limit, 'M', 'H'))

            Logger.info('Finish to create column {}'.format(target_col_name))
            Logger.debug('Group amount: {}'.format(str(merge_df[target_col_name].value_counts())))

        dir_path = '../results/data/vocabularies/'

        csv_file_name = '{}_{}.csv'.format(
            str(merge_df.shape[0]),
            str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        )

        file_path = CreateVocabularies._create_folder_and_save(
            merge_df,
            dir_path,
            csv_file_name,
            'save file with additional columns of PT groups')

        return file_path

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
            else:
                Logger.info('missing user id on mapping csv: ' + str(user_row['USER_SLCTD_ID']))

        Logger.info('High trait value users: found: {}, potential: {}'.format(
            str(len(user_id_high_percentile_list)),
                str(len(user_name_high_percentile_list))))

        Logger.info('Low trait value users: found: {}, potential: {}'.format(
            str(len(user_id_low_percentile_list)),
                str(len(user_name_low_percentile_list))))

        Logger.info('')

        # item id per group
        item_id_high_percentile_list = list()
        item_id_low_percentile_list = list()

        # iterate over all item description
        # store all item id, which their buyer id in high/low groups
        for index, user_row in self.user_item_id_df.iterrows():

            if index % 1000 == 0:
                Logger.info('insert item id into groups: {}/{}'.format(str(index), str(self.user_item_id_df.shape[0])))

            if user_row['buyer_id'] in user_id_high_percentile_list:
                item_id_high_percentile_list.append(user_row['item_id'])
            elif user_row['buyer_id'] in user_id_low_percentile_list:
                item_id_low_percentile_list.append(user_row['item_id'])

        Logger.info('Potential Number of description in high group: {}'.format(str(len(item_id_high_percentile_list))))
        Logger.info('Potential Number of description in low group: {}'.format(str(len(item_id_low_percentile_list))))

        if self.vocabulary_method == 'documents':
            cur_vocabulary_low_list = list()
            cur_vocabulary_low_dict = dict()
            cur_vocabulary_high_dict = dict()
            cur_vocabulary_high_list = list()

            miss = 0
            except_num = 0
            for index, row in self.description_df.iterrows():
                if index % 1000 == 0:
                    Logger.info('insert descriptions into groups: {}/{}'.format(str(index), str(self.description_df.shape[0])))

                if row['item_id'] in item_id_high_percentile_list:      # item high percentile
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_high_list.append(row['description'])
                        cur_vocabulary_high_dict[str(row['item_id'])] = row['description']

                elif row['item_id'] in item_id_low_percentile_list:     # item low percentile
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_low_list.append(row['description'])
                        cur_vocabulary_low_dict[str(row['item_id'])] = row['description']
                else:
                    # Logger.info('missing item_id in groups: ' + str(row['item_id']))
                    try:
                        user_id = self.user_item_id_df.loc[self.user_item_id_df['item_id'] == row['item_id']]['buyer_id'].values[0]
                        # print(self.user_ebay_df.loc[self.user_ebay_df['USER_ID'] == user_id]['USER_SLCTD_ID'].values)
                    except:
                        print('except')
                        except_num +=1
                        pass

                    miss += 1

            Logger.info('missing documents: ' + str(miss))
            Logger.info('except num documents: ' + str(except_num))
            Logger.info('Number of Valid description in high group: ' + str(len(cur_vocabulary_high_list)))
            Logger.info('Number of Valid description in low group: ' + str(len(cur_vocabulary_low_list)))
            self.save_vocabulary('high_' + str(self.personality_trait) + '_' + str(len(cur_vocabulary_high_list)), cur_vocabulary_high_list, cur_vocabulary_high_dict)
            self.save_vocabulary('low_' + str(self.personality_trait) + '_' + str(len(cur_vocabulary_low_list)), cur_vocabulary_low_list, cur_vocabulary_low_dict)

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
        else:
            raise ValueError('unknown vocabulary method')
        return

    # split items into groups by vertical
    def vertical_split_item_into_groups(self):

        item_description_list = list(self.description_df['item_id'])
        price_group = self.vertical_item_id_df.groupby(['BSNS_VRTCL_NAME'])

        # iterate over each vertical group
        for vertical_name, group_df in price_group:
            self.create_vocabulary(vertical_name, group_df)
        return

    # split items into group by both vertical and traits
    def traits_vertical_split_item_into_groups(self):

        high_limit = 0.5 + (self.gap_value / 2)
        low_limit = 0.5 - (self.gap_value / 2)

        Logger.info('High limit: ' + str(high_limit) + ', ' + 'Low limit: ' + str(low_limit))
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
        # item id per group
        item_id_high_percentile_list = list()
        item_id_low_percentile_list = list()

        for index, user_row in self.user_ebay_df.iterrows():
            if user_row['USER_SLCTD_ID'] in user_name_high_percentile_list:
                user_id_high_percentile_list.append(user_row['USER_ID'])
            elif user_row['USER_SLCTD_ID'] in user_name_low_percentile_list:
                user_id_low_percentile_list.append(user_row['USER_ID'])

        # get items per group
        for index, user_row in self.user_item_id_df.iterrows():
            if user_row['buyer_id'] in user_id_high_percentile_list:
                item_id_high_percentile_list.append(user_row['item_id'])
            elif user_row['buyer_id'] in user_id_low_percentile_list:
                item_id_low_percentile_list.append(user_row['item_id'])

        Logger.info('Potential Number of description in high group: ' + str(len(item_id_high_percentile_list)))
        Logger.info('Potential Number of description in low group: ' + str(len(item_id_low_percentile_list)))

        if self.vocabulary_method == 'documents':
            cur_vocabulary_low_list = list()
            cur_vocabulary_high_list = list()
            count_miss_ver = 0
            miss_desc = 0
            for index, row in self.description_df.iterrows():

                cur_item_data = self.full_user_item_id_df[self.full_user_item_id_df['item_id'] == row['item_id']]
                if cur_item_data.iloc[0]['BSNS_VRTCL_NAME'] != self.vertical:
                    # Logger.info('Vertical is not matching: ' + str(cur_item_data['BSNS_VRTCL_NAME']))
                    count_miss_ver += 1
                    continue

                if row['item_id'] in item_id_high_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_high_list.append(row['description'])
                    else:
                        miss_desc += 1

                elif row['item_id'] in item_id_low_percentile_list:
                    if isinstance(row['description'], basestring):
                        cur_vocabulary_low_list.append(row['description'])
                    else:
                        miss_desc += 1

            Logger.info('Number of missing verticals: ' + str(count_miss_ver))
            Logger.info('Numner missing descriptions: ' + str(miss_desc))

            Logger.info('Number of Valid description in high group: ' + str(len(cur_vocabulary_high_list)))
            Logger.info('Number of Valid description in low group: ' + str(len(cur_vocabulary_low_list)))

            self.save_vocabulary(str(self.vertical) + '_high_' + str(self.personality_trait) + '_amount_' + str(len(cur_vocabulary_high_list)), cur_vocabulary_high_list)
            self.save_vocabulary(str(self.vertical) + '_low_' + str(self.personality_trait) + '_amount_' + str(len(cur_vocabulary_low_list)), cur_vocabulary_low_list)
        else:
            raise('aggregation method is not defined')
        return

    # find all relevant text per group
    def create_vocabulary(self, vertical_name, group_df):

        Logger.info("Vartical name: " + str(vertical_name))
        item_id_list = group_df['item_id'].tolist()  # item id's of vertical
        Logger.info("Vartical size: " + str(group_df.shape[0]))

        # self.description_df['item_id'].astype(int)
        # get product description we have in the current vertical
        found_df = self.description_df.loc[self.description_df['item_id'].isin(item_id_list)]

        Logger.info("Description found: " + str(found_df.shape[0]))
        Logger.info("Total Description: " + str(group_df.shape[0]))
        Logger.info("Fraction Description: " + str(float(found_df.shape[0]) / float(group_df.shape[0])))

        # aggregate all description per group
        if self.vocabulary_method == 'documents':
            cur_vocabulary_list = list()
            for index, row in found_df.iterrows():
                if isinstance(row['description'], basestring):
                    cur_vocabulary_list.append(row['description'])
            Logger.info("Total descriptions: " + str(len(cur_vocabulary_list)))
            self.save_vocabulary(str(vertical_name), cur_vocabulary_list)

        elif self.vocabulary_method == 'aggregation':
            cur_vocabulary = ''                     # all words per group (vertical/traits)
            for index, row in found_df.iterrows():
                if isinstance(row['description'], basestring):
                    cur_vocabulary += row['description']
                    cur_vocabulary += ' '

            Logger.info("Total words: " + str(len(cur_vocabulary)))
            self.save_vocabulary(str(vertical_name), cur_vocabulary)
        return

    # save vocabulary per vertical
    def save_vocabulary(self, vertical_name, cur_vocabulary, cur_vocabulary_dict):

        directory_output = '../results/vocabulary/'

        group_file_name = '{}{}_{}_{}.txt'.format(
            str(self.dir_vocabulary_name),
            str(self.vocabulary_method),
            str(vertical_name),
            str(self.cur_time)
        )

        with open(group_file_name, 'wb') as fp:
            pickle.dump(cur_vocabulary, fp)

        Logger.info("Save file: {}".format(str(group_file_name)))
        Logger.info("")

        group_file_name = '{}{}_{}_{}_dict.txt'.format(
            str(self.dir_vocabulary_name),
            str(self.vocabulary_method),
            str(vertical_name),
            str(self.cur_time)
        )

        with open(group_file_name, 'wb') as fp:
            pickle.dump(cur_vocabulary_dict, fp)

        Logger.info("Save file: " + str(group_file_name))
        Logger.info("")
        return


def main(merge_df):

    # init class
    create_vocabularies_obj = CreateVocabularies(merge_df)

    create_vocabularies_obj.init_debug_log()                    # init log file
    create_vocabularies_obj.check_input()                       # check if arguments are valid
    create_vocabularies_obj.load_descriptions()                 # load description file
    create_vocabularies_obj.create_vocabulary_by_method()       # build model in regard to vertical/trait


if __name__ == '__main__':

    """
    description_file = '../results/data/truncate_description/7537_2018-08-02 11:12:34.csv'
    participants_survey_data = '../results/BFI_results/participant_bfi_score_check_duplication/clean_participant_958_2018-06-17 08:58:21.csv'
    participants_ebay_mapping_file = '../data/participant_data/1425 users input/personality_valid_users.csv'
    participants_purchase_history = '../data/participant_data/1425 users input/Purchase History format item_id.csv'
    """
    raise EnvironmentError('please run script from run_pre_processing.py')

    merge_df_path = '../results/data/POS/5485_2018-08-04 18:03:21.csv'
    main(merge_df_path)

