import pandas as pd
from time import gmtime, strftime
from utils.logger import Logger


class MergeDataSet:
    """
    Merge all data sets into one DF - for successive filtering
    build and save data-frame with all history purchase with all the features after merge the files
    """
    def __init__(self, description_file, user_purchase, user_bfi_score, map_user_name_id):
        """
        :param description_file: item with their clean text descriptions
        :param user_purchase: user purchase data
        :param user_bfi_score: user with their trait values and percentiles
        :param map_user_name_id: mapping of user_id - user_name
        """
        self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_dir = 'log/'
        self.verbose_flag = True

        self.description_file = description_file    # item id with description
        self.user_purchase = user_purchase          # all transaction (purchase) history
        self.user_bfi_score = user_bfi_score        # user with BFI score
        self.map_user_name_id = map_user_name_id    # map of user_name - user_id

        self.description_df = pd.DataFrame()        # item id with description
        self.user_purchase_df = pd.DataFrame()      # all transaction (purchase) history
        self.user_bfi_score_df = pd.DataFrame()         # user with BFI score
        self.map_user_name_id_df = pd.DataFrame()      # map user_name - user_id

    def init_debug_log(self):
        """ create a Logger object and save log in a file """

        file_prefix = 'merge_data_sets_wow'
        log_file_name = 'log/{}_{}.log'.format(file_prefix, self.time)
        Logger.set_handlers('MergeDataSet', log_file_name, level='debug')

    def run_merge_operation(self):
        """ merge data sets into one DF, store in csv """

        self.init_debug_log()
        self._load_data_sets()
        self._check_valid_input()
        intermediate_df = self._merge_all_df()
        self._save_df(intermediate_df)

    def _load_data_sets(self):
        """ load from csv to pandas DF all data """
        self.description_df = pd.read_csv(self.description_file)        # items and their descriptions
        self.user_purchase_df = pd.read_csv(self.user_purchase)         # participants survey data
        self.user_bfi_score_df = pd.read_csv(self.user_bfi_score)           # user with BFI score
        self.map_user_name_id_df = pd.read_csv(self.map_user_name_id)      # user with BFI score

        self.description_df['item_id'] = self.description_df['item_id'].astype(int)
        self.user_purchase_df['item_id'] = self.user_purchase_df['item_id'].astype(int)
        self.user_bfi_score_df['eBay site user name'] = self.user_bfi_score_df['eBay site user name'].str.lower()
        self.map_user_name_id_df['USER_SLCTD_ID'] = self.map_user_name_id_df['USER_SLCTD_ID'].str.lower()

    def _check_valid_input(self):
        """
        check df data is valid
        1. no duplication on id
        """
        if not pd.Series(self.user_bfi_score_df['eBay site user name']).is_unique:
            print(set(self.user_bfi_score_df['eBay site user name']) - set(self.user_bfi_score_df['eBay site user name'].unique()))
            raise ValueError('ebay user name must be unique - join will be wrong')

    def _merge_all_df(self):
        """ merge df into one """
        Logger.debug('1: Merge description_df-user_purchase_df')
        Logger.debug('L: description_df shape: ' + str(self.description_df.shape))
        Logger.debug('R: user_purchase_df shape: ' + str(self.user_purchase_df.shape))
        intermediate_df = self.description_df.merge(
            self.user_purchase_df,
            left_on='item_id',
            right_on='item_id',
            how='inner')
        Logger.debug('Result: Merge 1 shape: ' + str(intermediate_df.shape))
        Logger.debug('')

        Logger.debug('2: Merge description_df-user_purchase_df-map_user_name_id_df')
        Logger.debug('L: intermediate_df shape: ' + str(intermediate_df.shape))
        Logger.debug('R: map_user_name_id_df shape: ' + str(self.map_user_name_id_df.shape))
        intermediate_df = intermediate_df.merge(
            self.map_user_name_id_df,
            left_on='buyer_id',
            right_on='USER_ID',
            how='inner')
        Logger.debug('Result: Merge 2 shape: ' + str(intermediate_df.shape))
        Logger.debug('')

        Logger.debug('3: Merge description_df-user_purchase_df-map_user_name_id_df-user_bfi_score_df')
        Logger.debug('L: intermediate_df shape: ' + str(intermediate_df.shape))
        Logger.debug('R: user_bfi_score_df shape: ' + str(self.user_bfi_score_df.shape))
        intermediate_df = intermediate_df.merge(
            self.user_bfi_score_df,
            left_on='USER_SLCTD_ID',
            right_on='eBay site user name',
            how='inner')
        Logger.debug('Result: Merge 3 shape: ' + str(intermediate_df.shape))
        Logger.debug('')
        return intermediate_df

    def _save_df(self, intermediate_df):

        file_path = './results/data/merge_df_shape_{}_{}_time_{}.csv'.format(
            str(intermediate_df.shape[0]),
            str(intermediate_df.shape[1]),
            str(self.time)
        )
        intermediate_df.to_csv(file_path, index=False)
        Logger.debug('save csv file: ' + str(file_path))


def main():

    # currently, before and filtering
    description_file = './data/descriptions_data/1425 users input/merge_20048_time_2018-06-13 11:07:46.csv'
    # description_path = './data/descriptions_data/1425 users input/clean_12902.csv'        # after filtering

    user_purchase = './data/participant_data/1425 users input/Purchase History format item_id.csv'
    map_user_name_id = './data/participant_data/1425 users input/personality_valid_users.csv'           # insert new one (check Kira is missing e.g.)
    # user_bfi_score = './data/participant_data/1425 users input/users_with_bfi_score_amount_1425.csv'
    # user_bfi_score = './results/BFI_results//participant_bfi_score_check_duplication/clean_participant_987_2018-06-20 15:36:11.csv'    # remove threshold 0.5, length below 5
    # user_bfi_score = './results/BFI_results//participant_bfi_score_check_duplication/clean_participant_961_2018-06-20 16:46:36.csv'
    user_bfi_score = './results/BFI_results/participant_bfi_score_check_duplication/clean_participant_985_2018-06-20 16:50:15.csv'
    # user_bfi_score = './results/BFI_results//participant_bfi_score_check_duplication/clean_participant_1080_2018-06-20 15:32:02.csv'

    merge_data_sets = MergeDataSet(description_file, user_purchase, user_bfi_score, map_user_name_id)
    merge_data_sets.run_merge_operation()


if __name__ == '__main__':
    main()
