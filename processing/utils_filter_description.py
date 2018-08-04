# from __future__ import print_function
import os
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
    def filter_descriptions(merge_df, log_file_name, level='info'):
        """ main function clean description """

        Logger.set_handlers('FilterDescription', log_file_name, level=level)

        description_df = pd.read_csv(merge_df)
        # description_df = description_df.head(2000) - for debugging

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
                    Logger.info('parse language desc number: ' + str(idx) + ' / ' + str(description_df.shape[0]))
                if DROP_MAX or DROP_MIN:
                    desc = row['description'].decode("utf8")
                    description_df.at[idx, 'description_length'] = len(desc.split())
                if DROP_NON_ENGLISH:
                    detect_obj = detect_langs(desc)
                    if detect_obj[0].lang == 'en':
                        description_df.at[idx, 'en_bool'] = 1
                    else:
                        description_df.at[idx, 'en_bool'] = 0
            except Exception as e:
                print('exception found: ' + str(e))

        # before clean description store histogram of their length
        plt.style.use('seaborn-deep')
        plt.xlim(0, 1000)
        plt.hist(description_df['description_length'], bins=1000)
        plt.title('description length histogram')
        plt.savefig('../data/descriptions_data/1425 users input/histogram_description_length.png')
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