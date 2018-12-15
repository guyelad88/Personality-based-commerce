from time import gmtime, strftime

from utils_clean_description import CleanDescription
from utils.logger import Logger
from utils_balance_description import BalanceDescription
from utils_filter_description import FilterDescription
from remove_duplication import RemoveDuplication
from utils_pos import UtilsPOS
from create_vocabularies import CreateVocabularies
from calculate_kl import CalculateKL

import lexrank_config


class RunPreProcessing:
    """
    Merge all data sets into one DF - for successive filtering
    build and save data-frame with all history purchase with all the features after merge the files
    """
    def __init__(self):

        self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.configuration = lexrank_config.pre_processing_configs      # configuration of pre-process steps
        self.log_file_name = str()                              # allow access to the global log file name

    def init_debug_log(self):
        """ create a Logger object and save log in a file """

        file_prefix = 'run_pre_processing'
        self.log_file_name = '../log/{}_{}.log'.format(file_prefix, self.time)
        Logger.set_handlers('RunPreProcessing', self.log_file_name, level='debug')

    def run_pre_processing(self):
        """
        run pre-process steps
        main process:
        1. remove all fake/inconsistent users
        2. remove bad descriptions (non-english/na/too short/long) - FilterDescription
        3. balance data - truncate descriptions per user - BalanceDescription
        4. NLP pre-process steps - tokenizer/POS - UtilsPOS
        """

        # main argument: output of merge_data_sets df
        # merge_df_path = '../results/data/merge_user_purchase_description/13336_88_time_2018-08-04 17:50:30.csv'
        merge_df_path = '../results/data/merge_user_purchase_description/12377_88_time_2018-08-12 07:23:36.csv'

        Logger.info('start pre-process global method')

        """
        if self.configuration['pre_process_description']:

            Logger.info('Pre process descriptions')
            merge_df_path = CleanDescription._clean_description(
                merge_df_path=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')

        """
        """
        merge_df_path = '../results/data/clean_description/12377_2018-08-28 12:59:46.csv' # new method remove short sentences
        # merge_df_path = '../results/data/clean_description/12377_2018-08-21 09:08:14.csv'
        if self.configuration['filter_description']:
            Logger.info('filter description (language, length, etc..)')

            # merge_df_path = '../data/descriptions_data/1425 users input/merge_20048_time_2018-06-13 11:07:46.csv'
            merge_df_path = FilterDescription.filter_descriptions(
                merge_df=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')

        if self.configuration['remove_row_duplication']:
            Logger.info('remove entire row duplication')
            merge_df_path = RemoveDuplication.remove_duplication(
                merge_df_path=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')

        # merge_df_path = '../results/data/RemoveDuplication/9696_2018-08-12 07:38:32.csv'
        # balance data, truncate maximum number of description per user
        if self.configuration['balance_description_per_user']:

            Logger.info('balance number of description per user')
            merge_df_path = BalanceDescription.truncate_description_per_user_merged(
                merge_df_path=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')

        # merge_df_path = '../results/data/truncate_description/1330_2018-08-16 08:51:50.csv'
        if self.configuration['POS_filter']:

            Logger.info('add POS description column')
            merge_df_path = UtilsPOS.convert_to_pos(
                merge_df_path=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')

        if self.configuration['analyze_PT_groups']:

            Logger.info('add personality trait group columns (L/M/H)')
            merge_df_path = CreateVocabularies.traits_split_item_into_groups(
                merge_df_path=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')

        Logger.info('final DF save to a file: {}'.format(merge_df_path))
        Logger.info('finish pre-processing')
        """
        # NLP methods
        # merge_df_path = '../results/data/vocabularies/4195_2018-08-21 19:49:42.csv'
        merge_df_path = '../results/data/vocabularies/4123_2018-08-28 13:35:17.csv'  # after new cleaning method

        if self.configuration['calculate_KL']:
            Logger.info('calculate KL between each trait groups')
            calc_obj = CalculateKL(
                merge_df_path=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')
            calc_obj.run_kl()


def main():
    merge_data_sets = RunPreProcessing()
    merge_data_sets.init_debug_log()
    merge_data_sets.run_pre_processing()


if __name__ == '__main__':
    main()
