# from __future__ import print_function
import os
import re
import string

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import gmtime, strftime
from langdetect import detect_langs
from utils.logger import Logger
import config

MAX_LENGTH = config.filter_description['MAX_LENGTH']
MIN_LENGTH = config.filter_description['MIN_LENGTH']
DROP_NA = config.filter_description['DROP_NA']
DROP_MIN = config.filter_description['DROP_MIN']
DROP_MAX = config.filter_description['DROP_MAX']
DROP_NON_ENGLISH = config.filter_description['DROP_NON_ENGLISH']
DROP_NON_ENGLISH_WORDS = config.filter_description['DROP_NON_ENGLISH_WORDS']


class FilterDescription:
    """
        clean description - remain description which fulfill the following condition:
            a. description written in english
            b. description length between min and max
            c. description doesn't contain na

        :return: save the clean description in path ./data/description_data/.../clean_#description.csv
    """
    def __init__(self):
        """
        :param description_file: file contain description with
        :param output_dir:
        """
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    @staticmethod
    def _clean_description(desc):
        """
        tokenize description
        """

        row_desc = ''.join([i if ord(i) < 128 else ' ' for i in desc])      # remain only ascii chars
        row_desc = ' '.join(re.split('[.,]', row_desc))                     # split word.word e.g

        word_level_tokenizer = nltk.word_tokenize(row_desc)                 # tokenize using NLTK
        return word_level_tokenizer

    @staticmethod
    def filter_descriptions(merge_df, log_file_name, level='info'):
        """
        main function clean description

        the order is important
        1. remove NA
        2. determine description language and remain only english words using NLTK tokenizer
        3. calculate description length

        filter rows which did not satisfy the conditions above

        """

        Logger.set_handlers('FilterDescription', log_file_name, level=level)

        description_df = pd.read_csv(merge_df)
        # description_df = description_df.head(20)    # for debugging

        with open('../english_words/words_alpha.txt') as word_file:
            valid_english_words = set(word_file.read().split())

        # drop na
        if DROP_NA:
            before_nan_row = description_df.shape[0]
            # description_df = description_df.dropna(how='any') - old version without merge file as input
            description_df = description_df[pd.notnull(description_df['description'])]
            FilterDescription.calc_lost_ratio(before_nan_row, description_df.shape[0], 'drop nan')

        description_df["en_bool"] = np.nan
        description_df["description_length"] = 0

        for idx, row in description_df.iterrows():
            try:
                if idx % 1000 == 0:
                    Logger.info('remove all non-english words: {} / {}'.format(str(idx), str(description_df.shape[0])))

                desc = row['description'].decode("utf8")

                nltk.word_tokenize(''.join([i if ord(i) < 128 else ' ' for i in desc]))
                if DROP_NON_ENGLISH:

                    detect_obj = detect_langs(desc)
                    if detect_obj[0].lang == 'en':
                        description_df.at[idx, 'en_bool'] = 1
                    else:
                        description_df.at[idx, 'en_bool'] = 0

                assert DROP_NON_ENGLISH_WORDS is True       # TODO: generalize this, here we clean the descriptions
                if DROP_NON_ENGLISH_WORDS:
                    desc = FilterDescription._clean_description(desc)
                    desc_english = [
                        word.lower()
                        for word in desc
                        if word.lower() in valid_english_words or word.lower() in string.punctuation or word.isdigit()
                    ]
                    """desc_non_english = [
                        word.lower()
                        for word in desc
                        if word.lower() not in valid_english_words and word.lower() not in string.punctuation and not word.isdigit()
                    ]

                    # Logger.info('{}'.format(desc_non_english))
                    # Logger.info('{} / {}'.format(len(desc_english), len(desc)))
                    """

                    # squash description to contain only words in english
                    description_df.at[idx, 'description'] = ' '.join(desc_english)

            except BaseException as e:
                print('exception found: {}'.format(str(e)))

        for idx, row in description_df.iterrows():
            try:
                if idx % 1000 == 0:
                    Logger.info('parse language desc number: {} / {}'.format(str(idx), str(description_df.shape[0])))

                if DROP_MAX or DROP_MIN:
                    desc = row['description'].decode("utf8")
                    description_df.at[idx, 'description_length'] = len(desc.split())

            except BaseException as e:
                print('exception found: {}'.format(str(e)))
                print(idx)
                print(row)

        # before clean description store histogram of their length

        plot_dir = '../results/pre-processing/filter_description/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = '{}_histogram_description_length.png'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        plot_path = '{}{}'.format(plot_dir, plot_path)
        plt.style.use('seaborn-deep')
        plt.hist(description_df['description_length'], bins=1000)
        plt.title('description length histogram')
        plt.savefig(plot_path)
        plt.close()

        # remove length threshold
        if DROP_MIN:
            before = description_df.shape[0]
            description_df = description_df[description_df['description_length'] >= MIN_LENGTH]
            FilterDescription.calc_lost_ratio(before, description_df.shape[0], 'too short')

        if DROP_MAX:
            before = description_df.shape[0]
            description_df = description_df[description_df['description_length'] <= MAX_LENGTH]
            FilterDescription.calc_lost_ratio(before, description_df.shape[0], 'too long')

        # remove language threshold
        if DROP_NON_ENGLISH:
            before = description_df.shape[0]
            description_df = description_df[description_df['en_bool'] > 0]      # remain english only
            FilterDescription.calc_lost_ratio(before, description_df.shape[0], 'non english descriptions')

        # description_df = description_df[['item_id', 'description']]

        dir_path = '../results/data/filter_description/'

        csv_file_name = '{}_{}.csv'.format(
            str(description_df.shape[0]),
            str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        )

        file_path = FilterDescription._create_folder_and_save(
            description_df,
            dir_path,
            csv_file_name,
            'save file after clean description')

        return file_path

    @staticmethod
    def _create_folder_and_save(df, dir_path, file_name, log_title):
        """ save output data  """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_full_path = '{}{}'.format(dir_path, file_name)
        df.to_csv(file_full_path, index=False)
        Logger.info('{}: {}'.format(log_title, str(file_name)))

        return file_full_path

    @staticmethod
    def calc_lost_ratio(before_size, after_size, reason):
        """ generic function to log the discarded proportion of the data"""
        ratio = float(after_size)/float(before_size)
        Logger.info('drop {} - {}, deleted ratio: {} % , reason: {}'.format(
            str(before_size),
            str(after_size),
            str(1 - round(ratio, 3)),
            str(reason)))


def main(description_file, output_dir):

    filter_desc_obj = FilterDescription(description_file, output_dir)
    filter_desc_obj.init_debug_log()
    filter_desc_obj.filter_descriptions()


if __name__ == '__main__':

    raise EnvironmentError('please run script from run_pre_processing.py')

    description_file = '../data/descriptions_data/1425 users input/merge_20048_time_2018-06-13 11:07:46.csv'
    output_dir = '../data/descriptions_data/1425 users input/'
    main(description_file, output_dir)