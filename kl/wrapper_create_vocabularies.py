import pandas as pd
from processing.create_vocabularies import CreateVocabularies


# create vocabularies by grouping factor
# for each vertical and traits
class CreateVocabulariesWrapper:

    def __init__(self, description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method,
                 verbose_flag, participants_survey_data=None, participants_ebay_mapping_file=None,
                 participants_purchase_history=None, personality_trait=None, vertical=None):

        self.description_file = description_file  # description file
        self.log_dir = log_dir  # log directory
        self.directory_output = directory_output  # save description texts
        self.split_method = split_method  # split texts into groups using this method
        self.vocabulary_method = vocabulary_method  # output vocabulary separate by item description/ merge all
        self.verbose_flag = verbose_flag  # print results in addition to log file
        self.gap_value = gap_value  # gap between low and high group - e.g 0.5 keep above .75 under .25

        self.participants_survey_data = participants_survey_data  # participants survey data
        self.participants_ebay_mapping_file = participants_ebay_mapping_file  # ueBay user name + user_id
        self.participants_purchase_history = participants_purchase_history  # history purchase - item_id + item_data (vertical, price, etc.)
        self.personality_trait_list = personality_trait  # personality traits to split text by
        self.vertical_list = vertical  # vertical to split by

        self.description_df = pd.DataFrame()  # contain all descriptions
        self.vertical_item_id_df = pd.DataFrame()  # load item id vertical connection

        # traits split method
        self.user_trait_df = pd.DataFrame()  # user's and their personality traits percentile
        self.user_ebay_df = pd.DataFrame()  # map eBay user name and his user_id
        self.user_item_id_df = pd.DataFrame()  # user and his items he bought
        self.full_user_item_id_df = pd.DataFrame()  # item with all purchase data

        self.item_description_dict = dict()  # dictionary contain id and html code
        self.item_text_dict = dict()  # dictionary contain id and text extracted from html code

        self.cur_time = str
        self.dir_vocabulary_name = str


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
        if self.split_method not in ['vertical', 'traits', 'traits_vertical']:
            raise('split method: ' + str(self.split_method) + ' is not defined')

        if self.vocabulary_method not in ['documents', 'aggregation']:
            raise('vocabulary method: ' + str(self.vocabulary_method) + ' is not defined')

        if self.gap_value > 1 or self.gap_value < 0:
            raise('gap value must be between zero to one')
        return

    def run_one_experiment(self):
        for outer_idx, trait in enumerate(self.personality_trait_list):
            for inner_idx, vertical in enumerate(self.vertical_list):
                create_vocabularies_obj = CreateVocabularies(description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method, verbose_flag,
                    participants_survey_data, participants_ebay_mapping_file, participants_purchase_history, trait, vertical, self.cur_time)
                # create_vocabularies_obj.init_debug_log()  # init log file
                create_vocabularies_obj.check_input()  # check if arguments are valid
                create_vocabularies_obj.load_descriptions()  # load description file
                create_vocabularies_obj.create_vocabulary_by_method()  # build model in regard to vertical/trait
        return


def main(description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method, verbose_flag,
         participants_survey_data, participants_ebay_mapping_file, participants_purchase_history, personality_trait_list, vertical_list):

    # init class
    create_vocabularies_obj = CreateVocabulariesWrapper(description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method, verbose_flag,
         participants_survey_data, participants_ebay_mapping_file, participants_purchase_history, personality_trait_list, vertical_list)

    create_vocabularies_obj.init_debug_log()                    # init log file
    create_vocabularies_obj.check_input()                       # check if arguments are valid
    create_vocabularies_obj.run_one_experiment()


if __name__ == '__main__':

    # item and hist description file

    description_file = 'descriptions/num_items_1552_2018-01-30 13:15:33.csv'
    log_dir = 'log/'
    directory_output = 'vocabulary/'
    vocabulary_method = 'documents'     # 'documents', 'aggregation'
    verbose_flag = True
    split_method = 'traits_vertical'     # 'vertical', 'traits', 'traits_vertical'
    gap_value = 0.5                     # must be a float number between zero to one

    # needed if split_method is traits
    participants_survey_data = '../data/participant_data/participant_threshold_20_features_extraction.csv'  # users with more than 20 purchases
    participants_ebay_mapping_file = '../data/participant_data/personality_valid_users.csv'
    participants_purchase_history = '../data/participant_data/personality_purchase_history.csv'
    personality_trait_list = ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']
    vertical_list = ['Fashion', 'Electronics', 'Collectibles', 'Home & Garden']

    main(description_file, log_dir, directory_output, split_method, gap_value, vocabulary_method, verbose_flag,
         participants_survey_data, participants_ebay_mapping_file, participants_purchase_history, personality_trait_list, vertical_list)
