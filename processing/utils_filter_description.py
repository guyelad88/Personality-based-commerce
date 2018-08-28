# from __future__ import print_function
import os
import re
import string

import nltk
import nltk.data
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
DROP_DUPLICATION = config.filter_description['DROP_DUPLICATION']

REMOVE_UNINFORMATIVE_WORDS = config.filter_description['FLAG_UNINFORMATIVE_WORDS']
UNINFORMATIVE_WORDS_BI_GRAM = config.filter_description['BI_GRAM_UNINFORMATIVE_WORDS']
UNINFORMATIVE_WORDS = config.filter_description['UNINFORMATIVE_WORDS']

VERTICAL = config.filter_description['VERTICAL']


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
        # row_desc = ' '.join(re.split('[.,]', row_desc))                     # split word.word e.g

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
        # description_df = description_df.head(200)    # for debugging
        # description_df = description_df.loc[description_df['buyer_id'] == 1240181844]
        # with open('../english_words/words_alpha.txt') as word_file:
        #     valid_english_words = set(word_file.read().split())

        # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # import nltk
        from nltk.corpus import words
        from nltk.corpus import wordnet
        # manywords = words.words() + wordnet.words()
        # valid_english_words = set(manywords)
        valid_english_words = set(nltk.corpus.words.words())

        # drop na
        if DROP_NA:
            before_nan_row = description_df.shape[0]
            # description_df = description_df.dropna(how='any') - old version without merge file as input
            description_df = description_df[pd.notnull(description_df['description'])]
            FilterDescription.calc_lost_ratio(before_nan_row, description_df.shape[0], 'drop nan')

        if VERTICAL is not None:

            before = description_df.shape[0]
            description_df = description_df[description_df['BSNS_VRTCL_NAME'] == VERTICAL]
            FilterDescription.calc_lost_ratio(before, description_df.shape[0], 'Vertical')
            description_df = description_df.reset_index()

        description_df["en_bool"] = np.nan
        description_df["description_length"] = 0

        # check if is written in english and remain only words in english
        valid_sentence = 0
        invalid_sentence = 0
        length_valid_sentence = list()
        length_invalid_sentence = list()
        sentence_per_description = list()
        words_per_sentence = list()

        for idx, row in description_df.iterrows():
            try:
                if idx % 200 == 0:
                    Logger.info('remove all non-english words: {} / {}'.format(str(idx), str(description_df.shape[0])))

                desc = row['description'].decode("utf8")

                nltk.word_tokenize(''.join([i if ord(i) < 128 else ' ' for i in desc]))

                if DROP_NON_ENGLISH:

                    detect_obj = detect_langs(desc)         # detect language
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
                        if word.lower() in valid_english_words
                           or word.lower() in string.punctuation
                           # or word.isdigit()
                           or word[-1:] == 's'
                           or word[-2:] in ['ed', 'er']
                           or word[-3:] == ['ing', 'day', 'est', 'ear', 'ite']
                           or word[-4:] == 'able'
                    ]

                    """desc_english = [
                        word.lower()
                        for word in desc
                    ]"""

                    """if detect_obj[0].lang == 'en':
                        for word in desc:
                            if word.lower() not in valid_english_words and word.lower() not in string.punctuation and word[-1:] != 's' and word[-2:] != ['ed', 'er'] and word[-3:] not in ['ing', 'day', 'est', 'ear', 'ite']:
                                print(word)"""

                    """
                    desc_english = [
                        word.lower()
                        for word in desc
                    ]"""
                    # squash description to contain only words in english
                    desc = ' '.join(desc_english)
                    description_df.at[idx, 'description'] = desc

                # remove uninformative sentence (by removing if contain uninformative words)
                assert REMOVE_UNINFORMATIVE_WORDS is True
                if REMOVE_UNINFORMATIVE_WORDS:
                    # TODO: histogram of number of sentences, num deleted sentence, sentence length, deleted per length
                    # description not ine english will deleted any way

                    # print('english bool: {}'.format(description_df.at[idx, 'en_bool']))

                    if description_df.at[idx, 'en_bool'] == 1:
                        important_desc = list()
                        # sentence_list = tokenizer.tokenize(desc)
                        sentence_list = re.split(r"\.|\?|\;|\!", desc)

                        for s_idx, sentence in enumerate(sentence_list):
                            # print('before: {}'.format(sentence_list[s_idx]))
                            sentence_list[s_idx] = ' '.join([w for w in sentence.split() if len(w) > 1])
                            # print('after: {}'.format(sentence_list[s_idx]))

                        # remain only if number of char are bigger than 15 and at least 3 words
                        sentence_list = [sen for sen in sentence_list if len(sen.split()) > 2 and len(sen) > 5]
                        sentence_per_description.append(len(sentence_list))

                        for sentence in sentence_list:
                            # print(sentence)
                            sen_flag = True
                            words_per_sentence.append(len(sentence.split()))

                            # sentence language detection and filtering
                            """detect_obj = detect_langs(sentence)  # detect language
                            if detect_obj[0].lang != 'en':
                                print('{}: {}'.format(sentence, detect_obj))
                                sen_flag = False"""

                            # check heuristic
                            # reject sentences with uni-gram "bad" words (e.g. ship, fees)
                            for word in sentence.split():
                                if word in UNINFORMATIVE_WORDS:
                                    sen_flag = False

                            # reject sentences with bi-gram "bad" words (e.g. business day, thank you)
                            if sen_flag:
                                for uninformative_word in UNINFORMATIVE_WORDS_BI_GRAM:
                                    if uninformative_word in sentence:
                                        sen_flag = False

                            if sen_flag:
                                valid_sentence += 1
                                # print('valid_sentence')
                                length_valid_sentence.append(len(sentence.split()))
                                important_desc.extend([word for word in sentence.split() if len(word) > 1])
                                important_desc.append('.')      # end of sentence
                            else:
                                invalid_sentence += 1
                                # print('invalid_sentence')
                                length_invalid_sentence.append(len(sentence.split()))

                        # squash description to contain only words in english
                        desc = ' '.join(important_desc)
                        # print(desc)
                        description_df.at[idx, 'description'] = desc

            except BaseException as e:
                print('exception found: {}'.format(str(e)))

        FilterDescription._statistic_over_informative_sentences(valid_sentence, invalid_sentence,
                                                                sentence_per_description, words_per_sentence,
                                                                length_valid_sentence, length_invalid_sentence)
        # calculate description length (only english words)
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
        plt.xlim(0, 1200)
        plt.hist(np.array(description_df['description_length']), bins=400)
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

        if DROP_DUPLICATION:
            before = description_df.shape[0]
            description_df = description_df.drop_duplicates(
                subset=['description', 'buyer_id'],
                keep='first',
                inplace=False)
            FilterDescription.calc_lost_ratio(before, description_df.shape[0], 'duplication on buyer_id and description')

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

    @staticmethod
    def _statistic_over_informative_sentences(valid_sentence, invalid_sentence, sentence_per_description,
                                              words_per_sentence, length_valid_sentence, length_invalid_sentence):

        # remove sentence that contain details about payments etc..
        Logger.info('Valid sentences: {} Invalid sentence: {}'.format(valid_sentence, invalid_sentence))
        Logger.info('Valid sentences ratio: {}'.format(
            round(float(valid_sentence)/float(valid_sentence + invalid_sentence), 3)))

        plot_dir = '../results/pre-processing/filter_description/'
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        plot_path = '{}_sentence_per_description.png'.format(cur_time)
        plot_path = '{}{}'.format(plot_dir, plot_path)
        plt.style.use('seaborn-deep')
        plt.xlim(0, max(sentence_per_description))
        plt.hist(np.array(sentence_per_description), bins=20)
        plt.title('sentence per description histogram')
        plt.savefig(plot_path)
        plt.close()

        plot_path = '{}_words_per_sentence.png'.format(cur_time)
        plot_path = '{}{}'.format(plot_dir, plot_path)
        plt.style.use('seaborn-deep')
        plt.xlim(0, max(words_per_sentence))
        plt.hist(np.array(words_per_sentence), bins=50)
        plt.title('words_per_sentence histogram')
        plt.savefig(plot_path)
        plt.close()

        plot_path = '{}_length_valid_sentence.png'.format(cur_time)
        plot_path = '{}{}'.format(plot_dir, plot_path)
        plt.style.use('seaborn-deep')
        plt.xlim(0, max(length_valid_sentence))
        plt.hist(np.array(length_valid_sentence), bins=50)
        plt.title('length_valid_sentence histogram')
        plt.savefig(plot_path)
        plt.close()

        plot_path = '{}_length_invalid_sentence.png'.format(cur_time)
        plot_path = '{}{}'.format(plot_dir, plot_path)
        plt.style.use('seaborn-deep')
        plt.xlim(0, max(length_invalid_sentence))
        plt.hist(np.array(length_invalid_sentence), bins=50)
        plt.title('length_invalid_sentence histogram')
        plt.savefig(plot_path)
        plt.close()


def main(description_file, output_dir):

    filter_desc_obj = FilterDescription(description_file, output_dir)
    filter_desc_obj.init_debug_log()
    filter_desc_obj.filter_descriptions()


if __name__ == '__main__':

    raise EnvironmentError('please run script from run_pre_processing.py')

    description_file = '../data/descriptions_data/1425 users input/merge_20048_time_2018-06-13 11:07:46.csv'
    output_dir = '../data/descriptions_data/1425 users input/'
    main(description_file, output_dir)