# from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from time import gmtime, strftime
from langdetect import detect_langs

MAX_LENGTH = 500
MIN_LENGTH = 5

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
    def __init__(self):
        self.verbose_flag = True
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_dir = 'log/'

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
        description_file = '../data/descriptions_data/1425 users input/merge_20048.csv'
        description_df = pd.read_csv(description_file)

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
        file_path = '../data/descriptions_data/1425 users input/clean_' + str(description_df.shape[0]) + '.csv'
        description_df.to_csv(file_path, encoding='utf-8', index=False)
        logging.info('save file: ' + str(file_path))

    def calc_lost_ratio(self, before_size, after_size, reason):
        """ generic function to log the discarded proportion of the data"""
        ratio = float(after_size)/float(before_size)
        logging.info('drop ' + str(before_size) + ' - ' + str(after_size) + ', deleted ratio: ' +
                     str(1-round(ratio, 3)) + '%' + ', reason: ' + str(reason))

    """
        def check_num_dup(self):

        # HTML clean
        desc_1 = '/Users/gelad/Personality-based-commerce/kl/descriptions_clean/num_items_8704_2018-04-29 06:53:42.csv'
        desc_2 = '/Users/gelad/Personality-based-commerce/kl/descriptions_clean/num_items_11344_2018-04-29 07:00:12.csv'

        purchase_id = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/purchase_id_70451.csv'        # only item_id column
        purchase_history = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/personality_purchase_history.csv'        # all purchase data contain item_id column
        purchase_history_new = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/Purchase History format item_id.csv'

        purchase_history_xltx = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/Purchase History.xltx'
        desc_1_df = pd.read_csv(desc_1)     # Arnon HTML 1
        desc_2_df = pd.read_csv(desc_2)     # Arnon HTML 2



        purchase_history_df = pd.read_csv(purchase_history)
        purchase_history_df_new = pd.read_csv(purchase_history_new)
        # purchase_history_df = pd.read_excel(purchase_history)
        p_id_df = pd.read_csv(purchase_id)

        desc_big_df = desc_1_df.append(desc_2_df, ignore_index=True)    # All Arnon HTML (1+2 parts)
        num_desc = desc_big_df.shape[0]
        desc_big_df.to_csv('/Users/gelad/Personality-based-commerce/data/descriptions_data/1425 users input/merge_' +
                           str(num_desc) + '.csv',
                           encoding='utf-8',
                           index=False)

        return

        item_id_purchase_history_df = purchase_history_df['item_id'].tolist()     # purchase history with items_id
        item_id_3 = p_id_df['3211273377'].tolist()                  # list of purchase history item_id I gave to Arnon
        item_id_html = desc_big_df['item_id'].tolist()              # all descriptions item id from Arnon file

        item_id_purchase_history_df.sort()
        item_id_html.sort()
        item_id_3.sort()

        # item_id_2 = [(i/1000000)*1000000 for i in item_id_2]
        # item_id_1 = [int(i) for i in item_id_1]

        for idx, row in desc_big_df.iterrows():     # HTML
            if str(row['item_id']).startswith('292413'):
                print('')
                print('')
                print(row['item_id'])
                print(row['description'])

        print('')
        print('******html******')
        print('')
        for idx, row in purchase_history_df.iterrows():  # history item_id
            if str(int(row['item_id'])).startswith('292413'):
                print('')
                print(row['item_id'])
                print(row['AUCT_TITL'])

        print('')
        print('******html new******')
        print('')
        for idx, row in purchase_history_df_new.iterrows():  # history item_id
            if str(int(row['item_id'])).startswith('292413'):
                print('')
                print(row['item_id'])
                print(float(row['item_id']))
                print(int(row['item_id']))
                print(row['AUCT_TITL'])
        return


        for idx, item_id in enumerate(item_id_html):    # html item_id
            if str(item_id).startswith('292413'):
                print(idx)
                print(int(idx))
                print(item_id)

        print('******')
        for idx, item_id in enumerate(item_id_purchase_history_df):  # history item_id
            if str(int(item_id)).startswith('292413'):
                print(idx)
                print(int(item_id))
                print(float(item_id))
        return

        hit = 0
        miss = 0
        for item_id in item_id_1:       # Arnon items
            if item_id in item_id_html:    # history item_id
                print(item_id)
                hit +=1
                # print('hit')
            else:
                miss += 1
                # print('miss')

        print('hit: ' + str(hit))
        print('miss: ' + str(miss))

        return
    """


def main():

    filter_desc_obj = FilterDescription()
    filter_desc_obj.init_debug_log()
    filter_desc_obj.filter_descriptions()


if __name__ == '__main__':
    main()