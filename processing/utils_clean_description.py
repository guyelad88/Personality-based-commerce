# from __future__ import print_function
import os
import re
import pandas as pd
from time import gmtime, strftime
from utils.logger import Logger


class CleanDescription:
    """
        clean description
        :argument: csv file with all description after beautiful-soup cleaning.
        :return: save the clean descriptions in path ../results/data/clean_description/*_cur_time.csv
    """
    def __init__(self):
        pass

    @staticmethod
    def _clean_description(merge_df_path, log_file_name, level='info'):
        """
        clean description mainly using regex
        :param merge_df_path: df contain all descriptions
        """

        # load data
        # update logger properties
        Logger.set_handlers('CleanDescription', log_file_name, level=level)

        Logger.info('start to clean descriptions after beautiful-soup')

        # items and their descriptions
        merge_df = pd.read_csv(merge_df_path)

        # main actions
        for idx, row in merge_df.iterrows():
            try:
                if idx % 1000 == 0:
                    Logger.info('remove all non-english words: {} / {}'.format(str(idx), str(merge_df.shape[0])))

                desc = row['description'].decode("utf8")
                clean_desc = CleanDescription.tokenize_text_one(desc,
                                                                normalize_digits=False,
                                                                replace_apos=True,
                                                                replace_quot=True,
                                                                replace_amp=True)

                # replace description text
                merge_df.loc[idx, 'description'] = clean_desc

            except BaseException as e:
                pass
                # print('exception found: {}'.format(str(e)))

        # save results
        dir_path = '../results/data/clean_description/'
        csv_file_name = '{}_{}.csv'.format(
            str(merge_df.shape[0]),
            str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        )

        file_path = CleanDescription._create_folder_and_save(
            merge_df,
            dir_path,
            csv_file_name,
            'save file after clean descriptions')

        return file_path

    @staticmethod
    def _create_folder_and_save(df, dir_path, file_name, log_title):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = '{}{}'.format(dir_path, file_name)
        df.to_csv(file_path, index=False)
        Logger.info('{}: {}'.format(log_title, str(file_name)))

        return file_path

    @staticmethod
    def tokenize_text_one(line, normalize_digits=False, replace_apos=False, replace_quot=False, replace_amp=False):
        """
        1. replace /n to .
        2. split words concatenate with camelCase style.
        3. split words concatenate with .:+ e.g.
        """
        words = []

        _APOS_IN = "&apos;"
        _APOS_IN2 = "&apos ;"
        _APOS_OUT = "\'"
        _QUOT_IN = "&quot;"
        _QUOT_IN2 = "&quot ;"
        _QUOT_OUT = "\""
        _AMP_IN = "&amp;"
        _AMP_IN2 = "&amp ;"
        _AMP_OUT = "&"

        if replace_apos:
            line = re.sub(_APOS_IN, _APOS_OUT, line)
            line = re.sub(_APOS_IN2, _APOS_OUT, line)
        if replace_quot:
            line = re.sub(_QUOT_IN, _QUOT_OUT, line)
            line = re.sub(_QUOT_IN2, _QUOT_OUT, line)
        if replace_amp:
            line = re.sub(_AMP_IN, _AMP_OUT, line)
            line = re.sub(_AMP_IN2, _AMP_OUT, line)

        _WORD_SPLIT = re.compile("([.,!?+\"\-<>:;)(])")
        _DIGIT_RE = re.compile(r"\d")
        _DIGIT_OUT = '#'

        # change new line \n to . (easy to recognize later)
        line = ". ".join(line.split("\n"))
        Logger.info('Replace HTML end of line with dot')

        # split by camelCase
        for fragment in line.strip().split(' '):
            split_camel_case = re.sub('(?!^)([A-Z][a-z]+)', r' \1', fragment).split()
            split_camel_case = ['{}.'.format(tok) if i != len(split_camel_case)-1 and tok[-1] != '.' else tok for i, tok in enumerate(split_camel_case)]
            for token in split_camel_case:
                if not token:
                    continue
                words.append(token)
        output = " ".join(words)

        Logger.info('Split words in camelCase format')

        # split words concatenate with .:+ e.g.
        words = list()
        for fragment in output.strip().lower().split(' '):
            for token in re.split(_WORD_SPLIT, fragment):
                if not token:
                    continue
                if normalize_digits:
                    token = re.sub(_DIGIT_RE, _DIGIT_OUT, token)
                words.append(token)
        output = " ".join(words)

        Logger.info('Split words concatenate with . , e.g.')

        return output