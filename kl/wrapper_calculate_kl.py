import sys
import csv
import logging
from processing.calculate_kl import CreateVocabularies


# input: two descriptions file represent all texts of specific traits/vertical
# output: KL divergence of language model + contribute words for KL metric
class WrapperCalculateKL:

    def __init__(self, description_file_p, description_file_q, log_dir, results_dir, vocabulary_method,
                 results_dir_title, verbose_flag):

        # arguments
        self.description_file_p = description_file_p    # description file
        self.description_file_q = description_file_q    # description file
        self.log_dir = log_dir                          # log directory
        self.results_dir = results_dir                  # result directory
        self.vocabulary_method = vocabulary_method      # cal klPost, klCalc
        self.results_dir_title = results_dir_title      # init name to results file
        self.verbose_flag = verbose_flag                # print results in addition to log file

        import ntpath
        self.file_name_p = ntpath.basename(self.description_file_p)[:-4].split('_')[1]
        self.file_name_q = ntpath.basename(self.description_file_q)[:-4].split('_')[1]

        self.top_k_words = 30           # present top words
        self.SMOOTHING_FACTOR = 1.0     # smoothing factor for calculate term contribution

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        csv.field_size_limit(sys.maxsize)

    # build log object
    def init_debug_log(self):
        import logging
        from time import gmtime, strftime

        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        lod_file_name = self.log_dir + 'calculate_kl_' + str(self.cur_time) + '.log'

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

        logging.info("start log program")

    def check_input(self):

        if self.vocabulary_method not in ['documents', 'aggregation']:
            raise ('vocabulary method: ' + str(self.vocabulary_method) + ' is not defined')

        return

    # load vocabularies regards to vocabulary method
    def run_kl_wrapper_trait_vertical(self):
        import os

        counter = 0
        import pandas as pd
        df_meta = pd.DataFrame(columns=['trait', 'vertical', 'high', 'low'])

        for root, dirnames, filenames in os.walk('/Users/sguyelad/PycharmProjects/Personality-based-commerce/kl/vocabulary/2018-02-25 14:36:29'):
            for dirname in dirnames:

                counter += 1
                inter_dir ='/Users/sguyelad/PycharmProjects/Personality-based-commerce/kl/vocabulary/2018-02-25 14:36:29/' + str(dirname)
                list = os.listdir(inter_dir)  # dir is your directory path
                assert 2 == len(list)

                description_file_p = inter_dir + '/' + str(list[0])
                description_file_q = inter_dir + '/' + str(list[1])

                # extract vertical and trait + check validation
                trait = list[0].split('_')[3]
                vertical = list[0].split('_')[1]
                if trait not in ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']:
                    raise('trait undefined ' + str(trait))
                if vertical not in ['Fashion', 'Electronics', 'Collectibles', 'Home & Garden']:
                    raise ('vertical undefined ' + str(vertical))

                # insert which distribution is high/low
                p_title = None
                q_title = None
                if 'high' in list[0]:
                    p_title = 'high'
                elif 'low' in list[0]:
                    p_title = 'low'

                if 'high' in list[1]:
                    q_title = 'high'
                elif 'low' in list[1]:
                    q_title = 'low'

                cur_results_dir_title = str(trait) + '_' + str(vertical) + self.results_dir_title
                create_vocabularies_obj = CreateVocabularies(description_file_p, description_file_q, log_dir, results_dir,
                                                             vocabulary_method, cur_results_dir_title, verbose_flag, trait, vertical, p_title, q_title)

                if counter == 1:
                    create_vocabularies_obj.init_debug_log()  # init log file

                logging.info('Start Iteration ' + str(counter) + ' - ' + str(str(trait) + '_' + str(vertical)))
                create_vocabularies_obj.check_input()  # check if arguments are valid
                create_vocabularies_obj.run_kl()  # contain all inner functions

                # (columns = ['trait', 'vertical', 'high', 'low'])
                assert p_title == 'high'
                df_meta = df_meta.append({
                    'trait': trait,
                    'vertical': vertical,
                    'high': create_vocabularies_obj.len_p,
                    'low': create_vocabularies_obj.len_q
                },
                ignore_index=True)

        df_meta.to_csv('results/table_amount.csv')
        return

    # load vocabularies regards to vocabulary method
    def run_kl_wrapper_trait(self):

        counter = 0

        description_file_p = self.description_file_p
        description_file_q = self.description_file_q

        # extract vertical and trait + check validation
        trait = description_file_p.split('_')[2][:-4]
        vertical = ''

        if trait not in ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']:
            raise ('trait undefined ' + str(trait))

        # insert which distribution is high/low
        p_title = None
        q_title = None
        if 'high' in description_file_p:
            p_title = 'high'
        elif 'low' in description_file_p:
            p_title = 'low'

        if 'high' in description_file_q:
            q_title = 'high'
        elif 'low' in description_file_q:
            q_title = 'low'

        cur_results_dir_title = str(trait) + self.results_dir_title
        create_vocabularies_obj = CreateVocabularies(description_file_p, description_file_q, log_dir, results_dir,
                                                     vocabulary_method, cur_results_dir_title, verbose_flag, trait,
                                                     vertical, p_title, q_title, ngram_range=(1,1))


        create_vocabularies_obj.init_debug_log()  # init log file

        logging.info('Start Iteration ' + str(counter) + ' - ' + str(str(trait)))
        create_vocabularies_obj.check_input()  # check if arguments are valid
        create_vocabularies_obj.run_kl()  # contain all inner functions
        return


def main(description_file_p, description_file_q, log_dir, results_dir, vocabulary_method, results_dir_title,
         verbose_flag):

    # init class
    create_vocabularies_obj = WrapperCalculateKL(description_file_p, description_file_q, log_dir, results_dir,
                                                 vocabulary_method, results_dir_title, verbose_flag)

    # create_vocabularies_obj.init_debug_log()                    # init log file
    create_vocabularies_obj.check_input()                       # check if arguments are valid

    # create_vocabularies_obj.run_kl_wrapper_trait_vertical()                            # contain all inner functions
    create_vocabularies_obj.run_kl_wrapper_trait()  # contain all inner functions


if __name__ == '__main__':

    # neuroticism
    description_file_p = './vocabulary/2018-02-01 12:55:36/documents_high_neuroticism.txt'
    description_file_q = './vocabulary/2018-02-01 12:55:36/documents_low_neuroticism.txt'

    # extraversion
    description_file_p = './vocabulary/2018-02-01 13:16:22/documents_high_extraversion.txt'
    description_file_q = './vocabulary/2018-02-01 13:16:22/documents_low_extraversion.txt'

    # agreeableness
    description_file_p = './vocabulary/2018-02-01 13:14:08/documents_high_agreeableness.txt'
    description_file_q = './vocabulary/2018-02-01 13:14:08/documents_low_agreeableness.txt'

    # openness
    description_file_p = './vocabulary/2018-02-01 13:18:42/documents_high_openness.txt'
    description_file_q = './vocabulary/2018-02-01 13:18:42/documents_low_openness.txt'

    # conscientiousness
    # description_file_p = './vocabulary/2018-02-01 13:18:42/documents_high_openness.txt'
    # description_file_q = './vocabulary/2018-02-01 13:18:42/documents_low_openness.txt'

    log_dir = 'log/'
    results_dir = 'results/'
    results_dir_title = '_05_gap_'
    verbose_flag = True
    vocabulary_method = 'documents'     # 'documents', 'aggregation'

    main(description_file_p, description_file_q, log_dir, results_dir, vocabulary_method, results_dir_title,
         verbose_flag)
