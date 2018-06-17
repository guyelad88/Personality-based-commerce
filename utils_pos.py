import logging
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import nltk

VALID_POS = ['RBS', 'RB', 'RBR', 'JJ', 'JJR', 'JJS']


class Utils:

    def __init__(self):

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_dir = 'log/'
        self.verbose_flag = True
        self.logging = None

    # build log object
    def init_debug_log(self):

        log_file_name = self.log_dir + 'utils_pos_' + str(self.cur_time) + '.log'

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
    def convert_to_pos(df_description_path, logging):

        logging.info('start to convert description to POS tagging')
        # items and their descriptions
        description_df = pd.read_csv(df_description_path)
        description_df = description_df[['item_id', 'description']]

        df_pos_str = pd.DataFrame(columns=['item_id', 'description'])
        df_pos_list = pd.DataFrame(columns=['item_id', 'description'])

        for index, row in description_df.iterrows():
            if index % 1000 == 0:
                logging.info('POS pares description number: ' + str(index) + '/' + str(description_df.shape[0]))

            row_desc = ''.join([i if ord(i) < 128 else ' ' for i in row['description']])
            word_level_tokenizer = nltk.word_tokenize(row_desc)
            tuple_pos = nltk.pos_tag(word_level_tokenizer)
            pos_desc_str, pos_desc_list = Utils._extract_pos(tuple_pos)

            df_pos_str = df_pos_str.append(
                {
                    'item_id': row['item_id'],
                    'description': pos_desc_str
                },
                ignore_index=True)

            df_pos_list = df_pos_list.append(
                {
                    'item_id': row['item_id'],
                    'description': pos_desc_list
                },
                ignore_index=True)

        from time import gmtime, strftime
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        pos_string_path = './data/descriptions_data/1425 users input/clean_pos_str_' + str(df_pos_str.shape[0]) \
                          + '_time_' + str(cur_time) + '.csv'
        pos_list_path = './data/descriptions_data/1425 users input/clean_pos_list_' + str(df_pos_list.shape[0]) \
                        + '_time_' + str(cur_time) + '.csv'

        df_pos_str.to_csv(pos_string_path, index=False)
        logging.info('save POS-description in string format: ' + str(pos_string_path))

        df_pos_list.to_csv(pos_list_path, index=False)
        logging.info('save POS-description in list format: ' + str(pos_list_path))

    @staticmethod
    def _extract_pos(pos_tuple_list):

        pos_desc_str = ''               # return 'pos_1 pos_2 ... pos_n'
        pos_desc_list = list()          # return [pos_1, pos_2, ... , pos_n]

        for pos_tuple in pos_tuple_list:
            if pos_tuple[1] in VALID_POS:
                # pos_desc_str += pos_tuple[1]
                pos_desc_str += pos_tuple[0]
                pos_desc_str += ' '
                # pos_desc_list.append(pos_tuple[1])
                pos_desc_list.append(pos_tuple[0])

        return pos_desc_str, pos_desc_list

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
    description_path = './data/descriptions_data/1425 users input/clean_balance_8951_2018-06-13 11:30:15.csv'

    utils_cls.convert_to_pos(description_path, utils_cls.logging)


if __name__ == '__main__':
    main()
