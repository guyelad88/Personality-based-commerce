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
    def convert_to_pos(df_description_path, output_path, filter_pos_flag, save_pos, logging):
        """
        manipulate data regards to their POS tags
        :param df_description_path: df contain all descriptions
        :param output_path: path to save data after filter and transformation
        :param filter_pos_flag: remain words only with a specific POS tags
        :param save_pos: save POS or words
        :param logging: log object
        :return:
        """

        logging.info('start to convert description to POS tagging')
        logging.info('filter_pos_flag: ' + str(filter_pos_flag))
        logging.info('save_pos: ' + str(save_pos))
        logging.info('df input path: ') + str(df_description_path)
        logging.info('output path: ' + str(output_path))

        # items and their descriptions
        description_df = pd.read_csv(df_description_path)
        description_df = description_df[['item_id', 'description']]

        df_pos_str = pd.DataFrame(columns=['item_id', 'description'])
        df_pos_list = pd.DataFrame(columns=['item_id', 'description'])

        for index, row in description_df.iterrows():
            if index % 1000 == 0:
                logging.info('POS pares description number: ' + str(index) + '/' + str(description_df.shape[0]))

            row_desc = ''.join([i if ord(i) < 128 else ' ' for i in row['description']])    # remain only ascii chars
            word_level_tokenizer = nltk.word_tokenize(row_desc)
            tuple_pos = nltk.pos_tag(word_level_tokenizer)
            pos_desc_str, pos_desc_list = Utils._extract_pos(tuple_pos, filter_pos_flag, save_pos)

            df_pos_str = df_pos_str.append(         # save str version
                {
                    'item_id': row['item_id'],
                    'description': pos_desc_str
                },
                ignore_index=True)

            df_pos_list = df_pos_list.append(       # save list version
                {
                    'item_id': row['item_id'],
                    'description': pos_desc_list
                },
                ignore_index=True)

        from time import gmtime, strftime
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        pos_string_path = output_path + 'clean_pos_str_' + str(df_pos_str.shape[0]) \
                          + '_filter_' + str(filter_pos_flag) + '_pos_' + str(save_pos) \
                          + '_time_' + str(cur_time) + '.csv'

        pos_list_path = output_path + 'clean_pos_list_' + str(df_pos_list.shape[0]) \
                        + '_filter_' + str(filter_pos_flag) + '_pos_' + str(save_pos) \
                        + '_time_' + str(cur_time) + '.csv'

        df_pos_str.to_csv(pos_string_path, index=False)
        logging.info('save POS-description in string format: ' + str(pos_string_path))

        df_pos_list.to_csv(pos_list_path, index=False)
        logging.info('save POS-description in list format: ' + str(pos_list_path))

    @staticmethod
    def _extract_pos(pos_tuple_list, filter_pos_flag, save_pos):

        if save_pos:
            tuple_idx = 1       # save POS
        else:
            tuple_idx = 0       # save words

        pos_desc_str = ''               # return 'pos_1 pos_2 ... pos_n'
        pos_desc_list = list()          # return [pos_1, pos_2, ... , pos_n]

        for pos_tuple in pos_tuple_list:
            if filter_pos_flag and pos_tuple[1] not in VALID_POS:   # flag is true and POS isn't in list
                continue

            pos_desc_str += pos_tuple[tuple_idx]
            pos_desc_str += ' '
            pos_desc_list.append(pos_tuple[tuple_idx])

        return pos_desc_str, pos_desc_list


def main():

    utils_cls = Utils()
    utils_cls.init_debug_log()
    description_path = './data/descriptions_data/1425 users input/clean_balance_8951_2018-06-13 11:30:15.csv'
    output_path = './data/descriptions_data/1425 users input/'
    filter_pos_flag = True
    save_pos = False
    utils_cls.convert_to_pos(description_path, output_path, filter_pos_flag, save_pos, utils_cls.logging)


if __name__ == '__main__':
    main()
