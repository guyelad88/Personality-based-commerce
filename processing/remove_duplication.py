import os

import pandas as pd
from time import gmtime, strftime
import config
from utils.logger import Logger


class RemoveDuplication:

    def __init__(self):
        pass

    @staticmethod
    def remove_duplication(merge_df_path, log_file_name, level='info'):
        """
        add columns to DF with the POS taggers
        :param merge_df_path: df contain all descriptions
        :return:
        """

        # update logger properties
        Logger.set_handlers('RemoveDuplication', log_file_name, level=level)

        Logger.info('start to remove duplication in DF')

        # items and their descriptions
        merge_df = pd.read_csv(merge_df_path)

        df_without_dup = merge_df.drop_duplicates(subset=None, keep='first', inplace=False)

        Logger.info('Number of row before detecting duplication: {}'.format(merge_df.shape[0]))
        Logger.info('Number of row after detecting duplication: {}'.format(df_without_dup.shape[0]))
        Logger.info('Removed rows: {}'.format(merge_df.shape[0]-df_without_dup.shape[0]))

        # '../data/descriptions_data/1425 users input/clean_balance_{}_{}.csv'
        dir_path = '../results/data/RemoveDuplication/'
        csv_file_name = '{}_{}.csv'.format(
            str(df_without_dup.shape[0]),
            str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        )

        file_path = RemoveDuplication._create_folder_and_save(
            df_without_dup,
            dir_path,
            csv_file_name,
            'save file without duplications')

        return file_path

    @staticmethod
    def _create_folder_and_save(df, dir_path, file_name, log_title):

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = '{}{}'.format(dir_path, file_name)
        df.to_csv(file_path, index=False)
        Logger.info('{}: {}'.format(log_title, str(file_name)))

        return file_path
