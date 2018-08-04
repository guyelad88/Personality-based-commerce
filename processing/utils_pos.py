import pandas as pd
import os
import nltk
from time import gmtime, strftime
import config
from utils.logger import Logger

VALID_POS = config.POS['VALID_POS']                 # remain only this POS
FILTER_POS_FLAG = config.POS['filter_pos_flag']     # extract only a specific POS tags (appears in the VALID_POS list)
SAVE_POS = config.POS['save_pos']                   # save POS or words


class UtilsPOS:

    def __init__(self):
        pass

    @staticmethod
    def convert_to_pos(merge_df_path, log_file_name, level='info'):
        """
        add columns to DF with the POS taggers
        :param merge_df_path: df contain all descriptions
        :return:
        """
        
        # update logger properties
        Logger.set_handlers('ConvertToPOS', log_file_name, level=level)

        Logger.info('start to convert description to POS tagging')
        Logger.info('list of valid POS: ' + str(config.POS['VALID_POS']))
        Logger.info('filter_pos_flag: ' + str(FILTER_POS_FLAG))
        Logger.info('save_pos: ' + str(SAVE_POS))
        Logger.info('df input path: ' + str(merge_df_path))

        # items and their descriptions
        merge_df = pd.read_csv(merge_df_path)

        # add new columns
        merge_df['description_POS_str'] = None
        merge_df['description_POS_filter_str'] = None
        merge_df['description_POS_list'] = None
        merge_df['description_POS_filter_list'] = None

        for index, row in merge_df.iterrows():
            if index % 1000 == 0:
                Logger.debug('POS pares description number: {} / {}'.format(
                    str(index),
                    str(merge_df.shape[0])))

            row_desc = ''.join([i if ord(i) < 128 else ' ' for i in row['description']])    # remain only ascii chars
            word_level_tokenizer = nltk.word_tokenize(row_desc)     # return list of words
            tuple_pos = nltk.pos_tag(word_level_tokenizer)          # (word, POS) tuples to all description words.

            pos_desc_str, pos_desc_list, pos_desc_str_filter, pos_desc_list_filter = UtilsPOS._extract_pos(
                tuple_pos)

            merge_df.at[index, 'description_POS_str'] = pos_desc_str
            merge_df.at[index, 'description_POS_filter_str'] = pos_desc_str_filter
            merge_df.at[index, 'description_POS_list'] = pos_desc_list
            merge_df.at[index, 'description_POS_filter_list'] = pos_desc_list_filter

        # '../data/descriptions_data/1425 users input/clean_balance_{}_{}.csv'
        dir_path = '../results/data/POS/'
        csv_file_name = '{}_{}.csv'.format(
            str(merge_df.shape[0]),
            str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        )

        file_path = UtilsPOS._create_folder_and_save(
            merge_df,
            dir_path,
            csv_file_name,
            'save file with POS and specific POS taggers')

        return file_path

    @staticmethod
    def _extract_pos(pos_tuple_list):
        """ return list/str of POS/words according to save_pos bool value """

        tuple_idx = 1 if SAVE_POS else 0    # 1 means to save words, 0 the POS (by the indexes of each tuple)

        pos_desc_str = ''               # return 'pos_1 pos_2 ... pos_n'
        pos_desc_list = list()          # return [pos_1, pos_2, ... , pos_n]

        pos_desc_str_filter = ''
        pos_desc_list_filter = list()

        for pos_tuple in pos_tuple_list:

            pos_desc_str += pos_tuple[tuple_idx]
            pos_desc_str += ' '
            pos_desc_list.append(pos_tuple[tuple_idx])

            if FILTER_POS_FLAG and pos_tuple[1] not in VALID_POS:       # flag is true and POS isn't in list
                continue

            pos_desc_str_filter += pos_tuple[tuple_idx]
            pos_desc_str_filter += ' '
            pos_desc_list_filter.append(pos_tuple[tuple_idx])

        return pos_desc_str, pos_desc_list, pos_desc_str_filter, pos_desc_list_filter

    @staticmethod
    def _create_folder_and_save(df, dir_path, file_name, log_title):

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = '{}{}'.format(dir_path, file_name)
        df.to_csv(file_path, index=False)
        Logger.info('{}: {}'.format(log_title, str(file_name)))

        return file_path


def main():

    utils_cls = UtilsPOS()
    description_path = '../data/descriptions_data/1425 users input/clean_balance_8951_2018-06-13 11:30:15.csv'
    output_path = '../data/descriptions_data/1425 users input/'
    filter_pos_flag = True      # extract only a specific POS tags (as appears in the VALID_POS list)
    save_pos = True             # save POS or words
    utils_cls.convert_to_pos(description_path, output_path, filter_pos_flag, save_pos)


if __name__ == '__main__':
    raise EnvironmentError('please run script from run_pre_processing.py')
    main()
