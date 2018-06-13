import logging
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


class Utils:

    def __init__(self):

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_dir = 'log/'
        self.verbose_flag = True
        self.logging = None

    # build log object
    def init_debug_log(self):

        log_file_name = self.log_dir + 'utils_' + str(self.cur_time) + '.log'

        # logging.getLogger().addHandler(logging.StreamHandler())
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
    def truncate_description_per_user():
        """
        limit the number of descriptions per user in data-set. (purpose - only the ones which pass the filtering).
        hopefully it will avoid deviation of hte output language model.
        """
        pass

    @staticmethod
    def plot_histogram_users_number_of_description(df_description_path, participants_purchase_history_path, logging,
                                                   max_desc=None):
        """
        plot number of description per user appears in the data-set (purpose - only the ones which pass the filtering)
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
        Utils._calculate_list_statistic(a, logging)
        Utils._plot_histogram(a, user_group.ngroups, len(item_list), '')

        logging.info('')
        logging.info('remove zeros from list - users without purchase')
        a = list(filter(lambda x: x != 0, a))
        Utils._calculate_list_statistic(a, logging)
        Utils._plot_histogram(a, user_group.ngroups, len(item_list), '_non_zeros')

        if max_desc is None:
            max_desc = int(round(np.percentile(a, 95)))

        logging.info('')
        logging.info('max description: ' + str(max_desc))

        a = list(filter(lambda x: x <= max_desc, a))
        Utils._calculate_list_statistic(a, logging)
        Utils._plot_histogram(a, user_group.ngroups, len(item_list), '_truncated', max_desc)

        # truncate descriptions
        indexes_to_drop = []

        # TODO iterate over desc_id description df
        exist_items = 0
        user_group = user_item_id_df.groupby('buyer_id')
        for user, group in user_group:
            logging.info('')
            logging.info('user: ' + str(user) + ', number of purchases group: ' + str(group.shape[0]))

            user_items = 0
            for idx, item_id in group['item_id'].items():
                if int(item_id) in item_dict:
                    exist_items += 1
                    user_items += 1
                    if user_items > max_desc:
                        index_drop = description_df[description_df.item_id == int(item_id)].index[0]
                        indexes_to_drop.append(index_drop)
                        logging.info('user: ' + str(user) + ', idx: ' + str(idx))
        print(exist_items)
        print(len(indexes_to_drop))
        print(description_df.shape)
        description_df.drop(description_df.index[indexes_to_drop], inplace=True)
        print(description_df.shape)
        description_df.to_csv('./data/descriptions_data/1425 users input/clean_balance' +
                              str(description_df.shape[0]) + '.csv')
        raise()
        pass

    @staticmethod
    def clean_specific_words_from_description(clean_numbers=True, clean_pos=True):
        if clean_numbers:
            Utils.clean_numbers()
        if clean_pos:
            Utils.clean_pos()
        pass

    @staticmethod
    def clean_numbers():
        pass

    @staticmethod
    def clean_pos():
        pass

    @staticmethod
    def _calculate_list_statistic(np_list, logging):
        logging.info('0.05: ' + str(round(np.percentile(np_list, 5), 3)))
        logging.info('q1: ' + str(round(np.percentile(np_list, 25), 3)))
        logging.info('median: ' + str(round(np.percentile(np_list, 50), 3)))
        logging.info('q3: ' + str(round(np.percentile(np_list, 75), 3)))
        logging.info('0.95: ' + str(round(np.percentile(np_list, 95), 3)))
        logging.info('0.99: ' + str(round(np.percentile(np_list, 99), 3)))

    @staticmethod
    def _plot_histogram(a, user_amount, desc_num, additional_str, bin=100):

        plt.style.use('seaborn-deep')
        # plt.xlim(0, 100)
        # plt.ylim(0, y_max)
        plt.hist(a, bins=bin)
        plt.title('truncated description per user')
        plt.savefig('./results/utils/' + 'histogram_user_description' +
                    '_user_' + str(user_amount) +
                    '_desc_' + str(desc_num) + additional_str + '.png')
        plt.close()


def main():

    utils_cls = Utils()
    utils_cls.init_debug_log()
    participants_purchase_history = './data/participant_data/1425 users input/Purchase History format item_id.csv'
    description_path = './data/descriptions_data/1425 users input/clean_13430.csv'

    utils_cls.plot_histogram_users_number_of_description(description_path,
                                                         participants_purchase_history,
                                                         utils_cls.logging)


    pass


if __name__ == '__main__':
    main()
