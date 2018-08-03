from time import gmtime, strftime
from utils.logger import Logger
from utils_balance_description import BalanceDescription
from utils_filter_description import FilterDescription
import config


class RunPreProcessing:
    """
    Merge all data sets into one DF - for successive filtering
    build and save data-frame with all history purchase with all the features after merge the files
    """
    def __init__(self):

        self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.verbose_flag = True
        self.configuration = config.pre_processing_configs      # configuration of pre-process steps
        self.log_file_name = str()                              # allow access to global log file name

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
        2. remove bad descriptions (non-english/na/too short/long)
        3. balance data - truncate descriptions per user
        4. NLP pre-process steps - tokenizer/POS
        """

        merge_df_path = '../results/data/merge_df_shape_13336_88_time_2018-08-01 20:43:24.csv'

        Logger.info('start pre-process global method')
        # clean users
        if self.configuration['remove_fake_users']:
            # TODO process already inside survey_pilot/extract_BFI_score.py
            pass

        if self.configuration['remove_duplication']:
            # TODO process already inside survey_pilot/extract_BFI_score.py
            pass

        if self.configuration['filter_description']:
            Logger.info('filter description (language, length, etc..)')

            # merge_df_path = '../data/descriptions_data/1425 users input/merge_20048_time_2018-06-13 11:07:46.csv'
            merge_df_path = FilterDescription.filter_descriptions(
                merge_df=merge_df_path,
                log_file_name=self.log_file_name,
                level='info')

        # update path of merge_df
        # balance data
        if self.configuration['balance_description_per_user']:

            Logger.info('balance number of description per user')
            # merge_df_path = '../results/data/merge_df_shape_13336_88_time_2018-08-01 20:43:24.csv'  # TODO update path
            merge_df_path = BalanceDescription.truncate_description_per_user_merged(
                merge_df_path=merge_df_path,
                log_file_name=self.log_file_name,
                level='info',
                max_desc=None)

        if self.configuration['POS_filter']:

            Logger.info('add POS description column')

        Logger.info('finish pre-processing')



        # NLP methods


def main():
    merge_data_sets = RunPreProcessing()
    merge_data_sets.init_debug_log()
    merge_data_sets.run_pre_processing()


if __name__ == '__main__':
    main()
