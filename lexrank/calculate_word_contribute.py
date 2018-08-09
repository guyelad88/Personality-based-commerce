import pandas as pd
import statistics
from os import listdir
from os.path import isfile, join

from utils.logger import Logger


class CalculateWordContribute:
    """  """
    def __init__(self, trait_word_contribute_folder, trait_relative_path_dict, personality_trait_dict, time):

        self.trait_word_contribute_folder = trait_word_contribute_folder    # folder contain all token weights
        self.trait_relative_path_dict = trait_relative_path_dict            # specific dir contain kl
        self.personality_trait_dict = personality_trait_dict                # 'H'/'L' to each personality
        self.cur_time = time

        self.meta_word_contribute = dict()
        self.meta_word_count = dict()               # number of word appearance
        self.meta_word_values_diff_trait = dict()   # word cont in different traits

        return

    # calculate word contribute to KL using merging all user personality trait value
    def calculate_user_total_word_contribute(self):
        Logger.info('')
        for cur_trait, trait_value in self.personality_trait_dict.iteritems():      # check high/low input

            Logger.info('Personality trait: ' + str(cur_trait) + ', Type: ' + str(trait_value))

            # build file path

            cur_file_path = self.trait_word_contribute_folder + self.trait_relative_path_dict[cur_trait]

            trait_file_suffix = [f for f in listdir(cur_file_path) if isfile(join(cur_file_path, f))]

            if trait_value == 'H':
                file_name = [s for s in trait_file_suffix if 'high' in s]
                assert len(file_name) == 1
                cur_file_path += file_name[0]
            elif trait_value == 'L':
                file_name = [s for s in trait_file_suffix if 'low' in s]
                assert len(file_name) == 1
                cur_file_path += file_name[0]
            else:
                raise ValueError('currently we do not support M values, only H/L ')

            # load excel file into df
            cur_trait_df = pd.read_excel(open(cur_file_path, 'rb'), sheet_name=0)
            Logger.info('num of words: ' + str(cur_trait_df.shape[0]))

            # normalize trait to 0-1 scale (min-max version)
            cur_trait_df['contribute'] = (cur_trait_df['contribute'] - cur_trait_df['contribute'].min()) / (cur_trait_df['contribute'].max() - cur_trait_df['contribute'].min())

            Logger.info('normalize ' + str(cur_trait) + ' trait to 0-1 scale')

            for index, cur_row in cur_trait_df.iterrows():
                cur_word = cur_row['Word']
                cur_cont = cur_row['contribute']

                # check word is a string
                if not isinstance(cur_word, basestring):
                    continue

                # check word first time seen
                if cur_word not in self.meta_word_contribute:
                    self.meta_word_contribute[cur_word] = 0.0
                    self.meta_word_count[cur_word] = 0
                    self.meta_word_values_diff_trait[cur_word] = list()

                # update word total cont (moving average of old+new)
                prev_cont = self.meta_word_contribute[cur_word]
                prev_amount = self.meta_word_count[cur_word]
                new_cont = (prev_cont*prev_amount + cur_cont*1.0)/(prev_amount+1)
                self.meta_word_contribute[cur_word] = new_cont
                self.meta_word_count[cur_word] += 1
                self.meta_word_values_diff_trait[cur_word].append(round(cur_cont, 3))

        normalize_word_contributr_flag = True
        if normalize_word_contributr_flag:
            Logger.info('normalize values after aggregate all trait values together')
            min_value = min(self.meta_word_contribute.values())
            max_value = max(self.meta_word_contribute.values())
            denominator = max_value - min_value
            for cur_word, cur_val in self.meta_word_contribute.iteritems():
                self.meta_word_contribute[cur_word] = (cur_val-min_value)/denominator

        Logger.info('')
        Logger.info('word mean values: ' + str(round(statistics.mean(self.meta_word_contribute.values()), 3)))
        Logger.info('word std values: ' + str(round(statistics.stdev(self.meta_word_contribute.values()), 3)))

        self._log_word_contribute()         # save to log (and print in console) top k words
        self._save_word_contribute()        # save word contribution in csv
        return

    # log top k most associated and unrelated words to user personality
    def _log_word_contribute(self):
        import operator
        list_word_contribute_sort = sorted(self.meta_word_contribute.items(), key=operator.itemgetter(1))
        list_word_contribute_sort.reverse()

        top_show = 30
        Logger.info('')
        Logger.info('log top k=' + str(top_show) + ' associated and unrelated words to user personality')
        Logger.info('')
        Logger.info('word most associated with user personality:')
        for w_i, word_cont_tuple in enumerate(list_word_contribute_sort):
            try:
                if w_i >= top_show:
                    break
                line = self.get_line(w_i, word_cont_tuple)
                Logger.info(line)
            except:
                print('dkfd')
                pass

        list_word_contribute_sort.reverse()
        Logger.info('')
        Logger.info('word most unrelated with user personality:')
        for w_i, word_cont_tuple in enumerate(list_word_contribute_sort):
            if w_i >= top_show:
                break
            line = self.get_line(w_i, word_cont_tuple)
            Logger.info(line)

    # helper function to print row in log
    def get_line(self, w_i, word_cont_tuple):
        line = str(w_i) + ' ' + str(word_cont_tuple[0]) + ': ' + str(round(word_cont_tuple[1], 3)) + \
               ', #trait appear: ' + str(self.meta_word_count[word_cont_tuple[0]]) + ', trait values: ' + \
                str(self.meta_word_values_diff_trait[word_cont_tuple[0]])
        return line

    def _save_word_contribute(self, save_all=False):
        """insert all token in ascending order regards to their contribution to user personality"""

        Logger.info('')
        if not save_all:
            Logger.info('SKIP!!!! - save all word contribution with additional data - flag set to false')
            return

        Logger.info('save all word contribution with additional data')
        import operator
        # sort in ascending contribution order
        list_word_contribute_sort = sorted(self.meta_word_contribute.items(), key=operator.itemgetter(1))
        list_word_contribute_sort.reverse()

        df = pd.DataFrame(columns=[['word', 'contribution', 'num_trait', 'trait_values']])

        cnt = 0
        # insert token one-by-one
        for w_i, word_cont_tuple in enumerate(list_word_contribute_sort):
            try:
                df = df.append({
                    'word': word_cont_tuple[0],
                    'contribution': word_cont_tuple[1],
                    'num_trait': self.meta_word_count[word_cont_tuple[0]],
                    'trait_values': str(self.meta_word_values_diff_trait[word_cont_tuple[0]])
                }, ignore_index=True)
                cnt += 1
                if cnt % 1000 == 0:
                    Logger.info('add token with informative data: ' + str(cnt) + ' / ' + str(len(list_word_contribute_sort)))
            except Exception as e:
                Logger.info('exception word: ' + str(word_cont_tuple))
                Logger.info(e)
                Logger.info(e.message)

        dir_name = '../results/lexrank/personality_word_contribution/'

        for t, val in self.personality_trait_dict.items():
            dir_name += t[:1].upper()
            dir_name += '_'
            dir_name += str(val)
            dir_name += '_'
        dir_name = dir_name[:-1]
        dir_name += '/'

        import os
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        file_path = dir_name + str(self.cur_time) + '.csv'

        df.to_csv(file_path, index=False)
        Logger.info('save all word contribution: ' + str(file_path))


def main():
    pass


if __name__ == '__main__':
    raise Exception('main is not support from here')