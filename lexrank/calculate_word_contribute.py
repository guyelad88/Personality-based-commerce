import logging
import pandas as pd


class CalculateWordContribute:
    def __init__(self, trait_word_contribute_folder, personality_trait_dict, logging):

        self.trait_word_contribute_folder = trait_word_contribute_folder
        self.personality_trait_dict = personality_trait_dict
        self.logging = logging

        self.meta_word_contribute = dict()
        self.meta_word_count = dict()               # number of word appearance
        self.meta_word_values_diff_trait = dict()   # word cont in different traits

        return

    # calculate word contribute to KL using merging all user personality trait value
    def calculate_user_total_word_contribute(self):
        self.logging.info('')
        for cur_trait, trait_value in self.personality_trait_dict.iteritems():

            self.logging.info('Personality trait: ' + str(cur_trait) + ', Type: ' + str(trait_value))

            # build file path
            cur_file_path = self.trait_word_contribute_folder + '/' + str(cur_trait) + '_'
            if trait_value == 'H':
                cur_file_path += 'high.xls'
            elif trait_value == 'L':
                cur_file_path += 'low.xls'
            else:
                raise('currently we do not support M values')

            # load excel file into df
            cur_trait_df = pd.read_excel(open(cur_file_path, 'rb'), sheet_name=0)
            self.logging.info('num of words: ' + str(cur_trait_df.shape[0]))

            # normalize trait to 0-1 scale (min-max version)
            cur_trait_df['contribute'] = (cur_trait_df['contribute'] - cur_trait_df['contribute'].min()) / (cur_trait_df['contribute'].max() - cur_trait_df['contribute'].min())
            # fix nan - cur_trait_df['contribute'] = (cur_trait_df['contribute'] - cur_trait_df['contribute'].mean()) / cur_trait_df['contribute'].std()

            self.logging.info('normalize' + str(cur_trait) + ' trait to 0-1 scale')

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
            self.logging.info('normalize values after aggregate all trait values together')
            min_value = min(self.meta_word_contribute.values())
            max_value = max(self.meta_word_contribute.values())
            denominator = max_value - min_value
            for cur_word, cur_val in self.meta_word_contribute.iteritems():
                self.meta_word_contribute[cur_word] = (cur_val-min_value)/denominator

        import statistics
        self.logging.info('')
        self.logging.info('word mean values: ' + str(round(statistics.mean(self.meta_word_contribute.values()), 3)))
        self.logging.info('word std values: ' + str(round(statistics.stdev(self.meta_word_contribute.values()), 3)))

        self.log_word_contribute()
        # save all term contribute in a file
        return

    # log top k most associated and unrelated words to user personality
    def log_word_contribute(self):
        import operator
        list_word_contribute_sort = sorted(self.meta_word_contribute.items(), key=operator.itemgetter(1))
        list_word_contribute_sort.reverse()

        top_show = 30
        self.logging.info('')
        self.logging.info('log top k=' + str(top_show) + ' associated and unrelated words to user personality')
        self.logging.info('')
        self.logging.info('word most associated with user personality:')
        for w_i, word_cont_tuple in enumerate(list_word_contribute_sort):
            if w_i >= top_show:
                break
            line = self.get_line(w_i, word_cont_tuple)
            self.logging.info(line)

        list_word_contribute_sort.reverse()
        self.logging.info('')
        self.logging.info('word most unrelated with user personality:')
        for w_i, word_cont_tuple in enumerate(list_word_contribute_sort):
            if w_i >= top_show:
                break
            line = self.get_line(w_i, word_cont_tuple)
            self.logging.info(line)
        return

    # helper function to print row in log
    def get_line(self, w_i, word_cont_tuple):
        line = str(w_i) + ' ' + str(word_cont_tuple[0]) + ': ' + str(round(word_cont_tuple[1], 3)) + \
               ', #trait appear: ' + str(self.meta_word_count[word_cont_tuple[0]]) + ', trait values: ' + \
                str(self.meta_word_values_diff_trait[word_cont_tuple[0]])
        return line


def main(corpus_path_file, log_dir, target_sentences, trait_word_contribute_folder, personality_trait_dict,
         summary_size, threshold):
    CalWordContObj = CalculateWordContribute(trait_word_contribute_folder, personality_trait_dict)


if __name__ == '__main__':
    # current user personality - 'H'\'L'\'M' (high\low\miss(or mean))
    raise('main is not support from here')
    personality_trait_dict = {
        'openness': 'H',
        'conscientiousness': 'L',
        'extraversion': 'H',
        'agreeableness': 'H',
        'neuroticism': 'H'
    }

    trait_word_contribute_folder = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/kl/results/all_words_contribute'