# from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from time import gmtime, strftime
from langdetect import detect_langs

MAX_LENGTH = 500
MIN_LENGTH = 15

DROP_NA = True
DROP_MIN = True
DROP_MAX = True
DROP_NON_ENGLISH = True


class FilterDescription:
    """
        clean description:
        a. remain description language == 'english'
        b. description length between min and max
        c. drop description contain na

        :return: save description ./data/description_data/.../clean_#description.csv
    """
    def __init__(self, description_file, output_dir):
        self.verbose_flag = True
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_dir = 'log/'
        self.description_file = description_file
        self.output_dir = output_dir

    def init_debug_log(self):

        lod_file_name = self.log_dir + 'filter_description_' + str(self.cur_time) + '.log'

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

    def filter_descriptions(self):
        """ main function clean description """
        description_df = pd.read_csv(self.description_file)

        # drop na
        if DROP_NA:
            before_nan_row = description_df.shape[0]
            description_df = description_df.dropna(how='any')
            self.calc_lost_ratio(before_nan_row, description_df.shape[0], 'drop nan')

        description_df["en_bool"] = np.nan
        description_df["description_length"] = 0

        for idx, row in description_df.iterrows():
            try:
                if idx % 1000 == 0:
                    logging.info('parse language desc number: ' + str(idx) + ' / ' + str(description_df.shape[0]))
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
            self.calc_lost_ratio(before, description_df.shape[0], 'too short')

        if DROP_MAX:
            before = description_df.shape[0]
            description_df = description_df[description_df['description_length'] <= MAX_LENGTH]
            self.calc_lost_ratio(before, description_df.shape[0], 'too long')

        # remove language threshold
        if DROP_NON_ENGLISH:
            before = description_df.shape[0]
            description_df = description_df[description_df['en_bool'] > 0]      # remain english only
            self.calc_lost_ratio(before, description_df.shape[0], 'non english descriptions')

        description_df = description_df[['item_id', 'description']]

        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = output_dir + 'clean_' + str(description_df.shape[0]) + '.csv'
        description_df.to_csv(file_path, encoding='utf-8', index=False)
        logging.info('save file: ' + str(file_path))

    def calc_lost_ratio(self, before_size, after_size, reason):
        """ generic function to log the discarded proportion of the data"""
        ratio = float(after_size)/float(before_size)
        logging.info('drop ' + str(before_size) + ' - ' + str(after_size) + ', deleted ratio: ' +
                     str(1-round(ratio, 3)) + '%' + ', reason: ' + str(reason))


def main(description_file, output_dir):

    filter_desc_obj = FilterDescription(description_file, output_dir)
    filter_desc_obj.init_debug_log()
    filter_desc_obj.filter_descriptions()


if __name__ == '__main__':
    description_file = '../data/descriptions_data/1425 users input/merge_20048.csv'
    output_dir = '../data/descriptions_data/1425 users input/'
    main(description_file, output_dir)