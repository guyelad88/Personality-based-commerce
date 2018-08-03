import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from utils.logger import Logger


class BalanceDescription:
    """
    separate utils functions:
        1. truncate number of description per user
        2. merge DF
    """
    def __init__(self):
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    @staticmethod
    def truncate_description_per_user_merged(merge_df_path, log_file_name, level='info', max_desc=None):
        """
        limit the number of descriptions per user in the data-set.
        1.
        plot number of description per user appears in the data-set
        """

        # update logger properties
        Logger.set_handlers('BalanceDescription', log_file_name, level=level)

        merge_df = pd.read_csv(merge_df_path)  # history purchase
        Logger.info('merged df shape: {}, {}'.format(str(merge_df.shape[0]), str(merge_df.shape[1])))

        # calculate description distribution before truncate action
        item_amount_per_users = merge_df['buyer_id'].value_counts().tolist()
        a = np.array(item_amount_per_users)
        BalanceDescription._calculate_list_statistic(a)
        BalanceDescription._plot_histogram(a, len(a), sum(a), '')

        # build histogram without users with zero description
        Logger.info('')
        Logger.info('remove zeros from list - users without purchase')
        a = list(filter(lambda x: x != 0, a))
        BalanceDescription._calculate_list_statistic(a)
        BalanceDescription._plot_histogram(a, len(a), sum(a), '_non_zeros')

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
        BalanceDescription._calculate_list_statistic(np.array(a))
        BalanceDescription._plot_histogram(a, len(a), sum(a), '_truncated', max_desc)

        # '../data/descriptions_data/1425 users input/clean_balance_{}_{}.csv'
        dir_path = '../results/data/truncate_description/'
        csv_file_name = '{}_{}.csv'.format(
            str(merge_df.shape[0]),
            str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        )

        file_path = BalanceDescription._create_folder_and_save(
            merge_df,
            dir_path,
            csv_file_name,
            'save file truncate maximum num description')

        return file_path

    @staticmethod
    def _create_folder_and_save(df, dir_path, file_name, log_title):

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = '{}{}'.format(dir_path, file_name)
        df.to_csv(file_path)
        Logger.info('{}: {}'.format(log_title, str(file_name)))

        return file_path

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
        """
        merge two description file into one
        :param csv_1_path:
        :param csv_2_path:
        :return: merged description file
        """
        description_df_1 = pd.read_csv(csv_1_path)
        description_df_1 = description_df_1[['item_id', 'description']]

        description_df_2 = pd.read_csv(csv_2_path)
        description_df_2 = description_df_2[['item_id', 'description']]

        description_df = description_df_1.append(description_df_2)

        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        file_path = './data/descriptions_data/1425 users input/merge_{}_time_{}.csv'.format(
            str(description_df.shape[0]),
            str(cur_time)
        )

        description_df.to_csv(file_path, index=False)


def main():

    # please run the class from run_pre_processing.py
    print('please run the class from run_pre_processing.py')

    utils_cls = BalanceDescription()
    merge_df_path = './results/data/merge_df_shape_13336_88_time_2018-08-01 20:43:24.csv'
    cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    file_prefix = 'truncate_description'
    log_file_name = 'log/{}_{}.log'.format(file_prefix, cur_time)

    utils_cls.truncate_description_per_user_merged(merge_df_path=merge_df_path,
                                                   log_file_name=log_file_name,
                                                   max_desc=None)

if __name__ == '__main__':
    main()
