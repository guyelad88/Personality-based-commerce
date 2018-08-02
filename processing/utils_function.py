import logging
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from utils.logger import Logger


class UtilsFunction:
    """
    separate utils functions:
        1. truncate number of description per user
        2. merge DF
    """
    def __init__(self):

        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_dir = 'log/'
        self.verbose_flag = True
        self.logging = None

    # build log object
    def init_debug_log(self):

        log_file_name = self.log_dir + 'utils_' + str(self.cur_time) + '.log'

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logging.basicConfig(filename=log_file_name,
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        # print result in addition to log file
        if self.verbose_flag:
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

        logging.info("start log program")
        self.logging = logging


    @staticmethod
    def truncate_description_per_user(df_description_path, participants_purchase_history_path, logging, max_desc=None):
        """
        limit the number of descriptions per user in the data-set. (purpose - only the ones which pass the filtering).
        plot number of description per user appears in the data-set
        """
        # items and their descriptions
        description_df = pd.read_csv(df_description_path)
        description_df = description_df[['item_id', 'description']]

        user_item_id_df = pd.read_csv(participants_purchase_history_path)  # history purchase
        user_item_id_df = user_item_id_df[['item_id', 'buyer_id']]

        logging.info('Potential items bought: ' + str(user_item_id_df.shape[0]))
        logging.info('Actual items description: ' + str(description_df.shape[0]))

        item_list = description_df['item_id'].tolist()
        item_dict = dict((el, 0) for el in item_list)       # easy to check if item in dict

        item_amount_per_users = list()  # amount of found item in list

        total_found = 0
        total_found_int = 0
        user_group = user_item_id_df.groupby('buyer_id')
        for user, group in user_group:
            logging.info('')
            logging.info('user: ' + str(user) + ', number of purchases group: ' + str(group.shape[0]))

            # check number of exist items in the Data
            exist_items = 0
            for item_id in group['item_id']:
                if item_id in item_dict:
                    exist_items += 1
                    total_found += 1

                elif int(item_id) in item_dict:
                    exist_items += 1
                    total_found_int += 1

            item_amount_per_users.append(exist_items)
            logging.info('items found: ' + str(exist_items) + '/' + str(group.shape[0]) + ', ' +
                         str(round(float(exist_items)/float(group.shape[0]), 4)*100) + '%')

        logging.info('total found: ' + str(total_found))
        logging.info('total found int: ' + str(total_found_int))
        logging.info(max(item_amount_per_users))
        a = np.array(item_amount_per_users)
        UtilsFunction._calculate_list_statistic(a, logging)
        UtilsFunction._plot_histogram(a, user_group.ngroups, len(item_list), logging, '')

        logging.info('')
        logging.info('remove zeros from list - users without purchase')
        a = list(filter(lambda x: x != 0, a))
        UtilsFunction._calculate_list_statistic(a, logging)
        UtilsFunction._plot_histogram(a, user_group.ngroups, len(item_list), logging, '_non_zeros')

        if max_desc is None:
            max_desc = int(round(np.percentile(a, 95)))

        logging.info('')
        logging.info('max description: ' + str(max_desc))

        a = list(filter(lambda x: x <= max_desc, a))
        UtilsFunction._calculate_list_statistic(a, logging)
        UtilsFunction._plot_histogram(a, user_group.ngroups, len(item_list), logging, '_truncated', max_desc)

        # truncate descriptions
        indexes_to_drop = []

        # TODO iterate over desc_id description df
        exist_items = 0
        user_group = user_item_id_df.groupby('buyer_id')
        for user, group in user_group:
            logging.info('')
            logging.info('user: ' + str(user) + ', number of purchases group: ' + str(group.shape[0]))

            user_items = 0
            item_deleted = 0
            for idx, item_id in group['item_id'].items():
                if int(item_id) in item_dict:
                    exist_items += 1
                    user_items += 1
                    if user_items > max_desc:
                        item_deleted += 1
                        index_drop = description_df[description_df.item_id == int(item_id)].index[0]
                        indexes_to_drop.append(index_drop)
                        # logging.info('user: ' + str(user) + ', idx: ' + str(idx))
            logging.info('deleted items: ' + str(item_deleted))

        logging.info('Number of exist items: ' + str(exist_items))
        logging.info('Number of drop descriptions: ' + str(len(indexes_to_drop)))
        logging.info('Number of description before dropping: ' + str(description_df.shape))

        description_df.drop(description_df.index[indexes_to_drop], inplace=True)

        logging.info('Number of description after dropping: ' + str(description_df.shape))

        csv_file_name = './data/descriptions_data/1425 users input/clean_balance_' + str(description_df.shape[0]) + \
                        '_' + str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + '.csv'

        description_df.to_csv(csv_file_name)
        logging.info('save file after balance: ' + str(csv_file_name))

    @staticmethod
    def truncate_description_per_user_merged(merge_df_path, log_file_name, max_desc=None):
        """
        limit the number of descriptions per user in the data-set.
        1.
        plot number of description per user appears in the data-set
        """

        # update logger properties
        Logger.set_handlers('TruncateDescription', log_file_name, level='info')

        merge_df = pd.read_csv(merge_df_path)  # history purchase
        Logger.info('merged df shape: {}, {}'.format(str(merge_df.shape[0]), str(merge_df.shape[1])))

        # calculate description distribution before truncate action
        item_amount_per_users = merge_df['buyer_id'].value_counts().tolist()
        a = np.array(item_amount_per_users)
        UtilsFunction._calculate_list_statistic(a)
        UtilsFunction._plot_histogram(a, len(a), sum(a), '')

        # build histogram without users with zero description
        Logger.info('')
        Logger.info('remove zeros from list - users without purchase')
        a = list(filter(lambda x: x != 0, a))
        UtilsFunction._calculate_list_statistic(a)
        UtilsFunction._plot_histogram(a, len(a), sum(a), '_non_zeros')

        # choose maximum number of description per user
        if max_desc is None:
            max_desc = int(round(np.percentile(a, 95)))
        Logger.info('maximum number of description per user: ' + str(max_desc))

        # truncate descriptions
        merge_df['truncate_description'] = True
        user_group = merge_df.groupby('buyer_id')
        for user, group in user_group:
            Logger.debug('')
            Logger.debug('user: {}, number of purchases: {}'.format(str(user), str(group.shape[0])))
            user_item_idx = 0
            for s_idx, row in group.iterrows():

                user_item_idx += 1
                if user_item_idx > max_desc:
                    merge_df.at[s_idx, 'truncate_description'] = True       # row will remove
                else:
                    merge_df.at[s_idx, 'truncate_description'] = False      # row will remain

        # filter redundant description
        Logger.info('merged df shape: {}, {}'.format(str(merge_df.shape[0]), str(merge_df.shape[1])))
        merge_df = merge_df[~merge_df.truncate_description]
        Logger.info('merged df shape: {}, {}'.format(str(merge_df.shape[0]), str(merge_df.shape[1])))

        # show distribution after filtering redundant description
        a = merge_df['buyer_id'].value_counts().tolist()
        UtilsFunction._calculate_list_statistic(np.array(a))
        UtilsFunction._plot_histogram(a, len(a), sum(a), '_truncated', max_desc)

        # '../data/descriptions_data/1425 users input/clean_balance_{}_{}.csv'
        dir_path = '../results/data/truncate_description/'
        csv_file_name = '{}_{}.csv'.format(
            str(merge_df.shape[0]),
            str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        )

        UtilsFunction._create_folder_and_save(merge_df, dir_path, csv_file_name, 'save file truncate maximum num description')

    @staticmethod
    def _create_folder_and_save(df, dir_path, file_name, log_title):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        df.to_csv('{}{}'.format(dir_path, file_name))
        Logger.info('{}: {}'.format(log_title, str(file_name)))

    @staticmethod
    def clean_specific_words_from_description(clean_numbers=True, clean_pos=True):
        if clean_numbers:
            UtilsFunction.clean_numbers()
        if clean_pos:
            UtilsFunction.clean_pos()
        pass

    @staticmethod
    def clean_numbers():
        pass

    @staticmethod
    def clean_pos():
        pass

    @staticmethod
    def _calculate_list_statistic(np_list):
        Logger.info('0.05: ' + str(round(np.percentile(np_list, 5), 3)))
        Logger.info('q1: ' + str(round(np.percentile(np_list, 25), 3)))
        Logger.info('median: ' + str(round(np.percentile(np_list, 50), 3)))
        Logger.info('q3: ' + str(round(np.percentile(np_list, 75), 3)))
        Logger.info('0.95: ' + str(round(np.percentile(np_list, 95), 3)))
        Logger.info('0.99: ' + str(round(np.percentile(np_list, 99), 3)))

    @staticmethod
    def _plot_histogram(a, user_amount, desc_num, additional_str, bin=100):
        """ generic file to plot histogram and save plot """

        plt.style.use('seaborn-deep')
        plt.hist(a, bins=bin)
        plt.title('description per user {}'.format(additional_str))

        plot_path = '../results/utils/HistogramUserDescriptionAmount_user_{}_desc_{}{}.png'.format(
            str(user_amount),
            str(desc_num),
            additional_str)

        plt.savefig(plot_path)
        plt.close()

        Logger.info('save histogram plot: ' + str(plot_path))

    @staticmethod
    def _merge_to_csv(csv_1_path, csv_2_path):

        description_df_1 = pd.read_csv(csv_1_path)
        description_df_1 = description_df_1[['item_id', 'description']]

        description_df_2 = pd.read_csv(csv_2_path)
        description_df_2 = description_df_2[['item_id', 'description']]

        description_df = description_df_1.append(description_df_2)

        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        description_df.to_csv('./data/descriptions_data/1425 users input/merge_' +
                              str(description_df.shape[0]) + '_time_' + str(cur_time) + '.csv', index=False)

        pass


def main():

    utils_cls = UtilsFunction()

    participants_purchase_history = './data/participant_data/1425 users input/Purchase History format item_id.csv'
    # description_path = './data/descriptions_data/1425 users input/clean_12902.csv'
    description_path = './data/descriptions_data/1425 users input/clean_balance_8646_2018-06-17 15:17:04.csv'
    merge_df_path = './results/data/merge_df_shape_13336_88_time_2018-08-01 20:43:24.csv'

    log_file_path = 'log/merge_data_sets_2018-06-20 11:41:43.log'
    cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    log_dir = 'log/'
    file_prefix = 'aaaaa'
    log_file_name = 'log/{}_{}.log'.format(file_prefix, cur_time)

    utils_cls.truncate_description_per_user_merged(merge_df_path=merge_df_path,
                                                   log_file_name=log_file_name,
                                                   max_desc=None)

    """utils_cls.truncate_description_per_user(description_path,
                                            participants_purchase_history,
                                            utils_cls.logging)"""

    # csv_1_path = './kl/descriptions_clean/num_items_11344_2018-06-13 09:09:14.csv'
    # csv_2_path = './kl/descriptions_clean/num_items_8704_2018-06-13 10:26:12.csv'
    # utils_cls._merge_to_csv(csv_1_path, csv_2_path)

    pass


if __name__ == '__main__':
    main()
