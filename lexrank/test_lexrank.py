import json
import math

import pandas as pd
from time import gmtime, strftime

import lexrank_config
from summarizer import LexRank
from calculate_word_contribute import CalculateWordContribute
from utils.logger import Logger


SUMMARY_SIZE = lexrank_config.test_lexrank['summary_size']
HTML_SUMMARY_SIZE = lexrank_config.test_lexrank['HTML_summary_size']

THRESHOLD = lexrank_config.test_lexrank['threshold']
DAMPING_FACTOR = lexrank_config.test_lexrank['damping_factor']
SUMMARIZATION_SIMILARITY_THRESHOLD = lexrank_config.test_lexrank['summarization_similarity_threshold']
TARGET_SENTENCES_LENGTH_MIN = lexrank_config.test_lexrank['target_sentences_length']['min']
TARGET_SENTENCES_LENGTH_MAX = lexrank_config.test_lexrank['target_sentences_length']['max']
CORPUS_SIZE = lexrank_config.test_lexrank['corpus_size']

PERSONALITY_WORD_FLAG = lexrank_config.test_lexrank['personality_word_flag']
RANDOM_WALK_FLAG = lexrank_config.test_lexrank['random_walk_flag']

PRODUCTS_IDS = lexrank_config.test_lexrank['products_ids']      # id's to run algorithm on

MULTI_DOCUMENT_SUMMARIZATION = lexrank_config.test_lexrank['multi_document_summarization']
LEX_RANK_ALGORITHM_VERSION = lexrank_config.test_lexrank['lex_rank_algorithm_version']
SUMMARIZATION_VERSION = lexrank_config.test_lexrank['summarization_version']
PERSONALITY_TRAIT_DICT = lexrank_config.test_lexrank['personality_trait_dict']

# data paths
CORPUS_PATH_FILE = lexrank_config.test_lexrank['corpus_path_file']
TARGET_ITEM_DESCRIPTION_FILE = lexrank_config.test_lexrank['target_item_description_file']
TRAIT_RELATIVE_PATH_DICT = lexrank_config.test_lexrank['trait_relative_path_dict']

log_dir = 'log/'
html_dir = '../results/lexrank/html/'


class WrapperLexRank:

    """
    Main function of the algorithm - run personalized-LexRank algorithm

    :argument
    summary_size: number of senetnces in the final summarization. 'max' or int
    threshold: min similarity between two node (sen.) in the graph, discarded edge if sim is smaller.
    damping_factor: prob. *not* to jump (low -> 'personalized' jump often occur)
    summarization_similarity_threshold = max similarity of two sentences in the final desc.
    target_sentences_length: {'min': , 'max': } - number of sentences in the input description.

    personality_trait_dict: user personality ('H'/'L' assign to each trait)

    corpus_size = 'max'
    personality_word_flag:          if combine
    random_walk_flag:               if combine "personalized" jump (matrix).
    multi_document_summarization:   summarization method for single/multi documents
    lex_rank_algorithm_version:     'personality-based-LexRank', 'vanilla-LexRank'
    summarization_version:          'top_relevant', 'Bollegata', 'Shahaf'

    corpus_path_file: file contain description to calculate idf from
    target_item_description_file: clean description (Amazon product) to run LexRank on them

    log_dir = 'log/'
    html_dir = 'html/'

    :raises

    :returns

    """
    def __init__(self):
        an = self
        self.input_attr_dict = vars(an)         # save attribute to show them later
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.log_file_name = None
        self.word_cont_dict = dict()    # word and correspond contribute value

    # build log object
    def init_debug_log(self):
        file_prefix = 'test_lexrank'
        self.log_file_name = '../log/{}_{}.log'.format(file_prefix, self.cur_time)
        Logger.set_handlers('TestLexRank', self.log_file_name, level='info')

    # extract corpus documents from a file
    @staticmethod
    def load_corpus_documents(corpus_file):

        '''
        :param corpus_file: path to file contain corpuse data (documents)
        :return: list of string, each str contain item description
        '''

        description_df = pd.read_csv(corpus_file)  # items and their descriptions
        description_df = description_df['description'].values.astype(str)     # description_df[['item_id', 'description']]
        documents = description_df.tolist()
        return documents

    # load item description to summarize
    @staticmethod
    def load_description_df():
        target_description_df = pd.read_excel(TARGET_ITEM_DESCRIPTION_FILE)  # items and their descriptions
        return target_description_df

    # check arguments are valid
    @staticmethod
    def check_input():

        for c_trait, user_value in PERSONALITY_TRAIT_DICT.iteritems():
            if user_value not in ['H', 'L', 'M']:
                raise ValueError('unknown trait value')

        if DAMPING_FACTOR < 0 or DAMPING_FACTOR > 1:
            raise ValueError('damping factor must be a float between 0 to 1')

        if SUMMARIZATION_SIMILARITY_THRESHOLD < 0 or SUMMARIZATION_SIMILARITY_THRESHOLD > 1:
            raise ValueError('summarization similarity threshold must be a float between 0 to 1')

        if LEX_RANK_ALGORITHM_VERSION not in ['vanilla-LexRank', 'personality-based-LexRank']:
            raise ValueError('unknown lex_rank_algorithm_version')

        if SUMMARIZATION_VERSION not in ['top_relevant', 'Bollegata', 'Shahaf']:
            raise ValueError('unknown summarization_version')

        if MULTI_DOCUMENT_SUMMARIZATION not in ['single', 'multi']:
            raise ValueError('unknown multi_document_summarization value (not single nor multi)')

        if CORPUS_SIZE != 'max' and not isinstance(CORPUS_SIZE, int):
            raise ValueError('unknown corpus size variable')

        if CORPUS_SIZE != 'max' and CORPUS_SIZE < 20:
            raise ValueError('too small corpus size')

        if LEX_RANK_ALGORITHM_VERSION == 'vanilla-LexRank' and DAMPING_FACTOR not in [0.15, 0.25, 0.5]:
            raise ValueError('for vanilla LexRank damping factor must be 0.15')

        if LEX_RANK_ALGORITHM_VERSION == 'personality-based-LexRank' and DAMPING_FACTOR not in [0.01, 0.1, 0.2, 0.3]:
            raise ValueError('for personality-based-LexRank damping factor must be in [0.01, 0.1, 0.2, 0.3]')

    # log class arguments
    def log_attribute_input(self):

        Logger.info('')
        Logger.info('Class arguments')
        for attr, attr_value in self.input_attr_dict.iteritems():
            Logger.info('Attribute: {}, Value: {}'.format(str(attr), str(attr_value)))
        Logger.info('')
        return

    # main function - run LexRank
    def test_lexrank(self):
        '''
        :return: run LexRank class with corpus, list of sentences to summary and configuration
        '''

        self.check_input()              # check inputs are valid
        self.log_attribute_input()      # log class arguments

        # calculate word contribute to current personality
        word_contibute_obj = CalculateWordContribute(TRAIT_RELATIVE_PATH_DICT,
                                                     PERSONALITY_TRAIT_DICT,
                                                     self.cur_time)
        word_contibute_obj.calculate_user_total_word_contribute()
        word_cont_dict = word_contibute_obj.meta_word_contribute        # personality word contribute

        # collect corpus data
        documents = self.load_corpus_documents(CORPUS_PATH_FILE)

        # load item description to summarize
        target_description_df = self.load_description_df()

        for index, cur_row_target_description in target_description_df.iterrows():
            # print(index)
            # print(cur_row_target_description['TITLE'])

            desc_id = cur_row_target_description['ID'].encode('ascii', 'ignore')

            if not isinstance(desc_id, str):
                if math.isnan(desc_id):
                    Logger.info('Load Nan Id row - Skip')
                    continue

            desc_title = cur_row_target_description['TITLE'].encode('ascii', 'ignore')
            target_sentences = json.loads(cur_row_target_description['DESCRIPTION'])
            target_sentences = [s.encode('ascii', 'ignore') for s in target_sentences]
            num_sentences = cur_row_target_description['LENGTH']

            # control amount of sentences
            if num_sentences > TARGET_SENTENCES_LENGTH_MAX or num_sentences < TARGET_SENTENCES_LENGTH_MIN:
                continue
            if desc_id not in PRODUCTS_IDS:         #  baby bandana: 'B0746GQ56P'. 'B06XXY79N1', 'B003MYYJD0'
                continue

            # define summarization length
            if SUMMARY_SIZE == 'max':
                summary_size = len(target_sentences)
            else:
                summary_size = SUMMARY_SIZE

            Logger.info('')
            Logger.info('Description ID: {}, Length: {}, Title: {}'.format(
                str(desc_id), str(num_sentences), str(desc_title))
            )
            Logger.info('item sentences: {}'.format(str(len(target_sentences))))

            # build LexRank class obj, calculate idf on entire corpus
            lxr = LexRank(
                documents,
                LEX_RANK_ALGORITHM_VERSION,
                SUMMARIZATION_VERSION,
                html_dir=html_dir,
                user_personality_dict=PERSONALITY_TRAIT_DICT,
                word_cont_dict=word_cont_dict,
                personality_word_flag=PERSONALITY_WORD_FLAG,        # similarity based personality
                random_walk_flag=RANDOM_WALK_FLAG,
                damping_factor=DAMPING_FACTOR,
                summarization_similarity_threshold=SUMMARIZATION_SIMILARITY_THRESHOLD,
                multi_document_summarization=MULTI_DOCUMENT_SUMMARIZATION,   # multi/single summarization
                cur_time=self.cur_time,
                desc_title=desc_title,
                desc_id=desc_id,
                corpus_size=CORPUS_SIZE,
                stopwords=None,                                     # STOPWORDS['en'],
                keep_numbers=False,
                keep_emails=False,
                include_new_words=True,
            )

            # build similarity matrix, power method
            summary, sorted_ix, lex_scores, description_summary_list, discarded_sentences_list = lxr.get_summary(
                target_sentences,
                summary_size,
                THRESHOLD,
                discretize=False)

            # write results into log file
            self.log_results(summary, sorted_ix, lex_scores, description_summary_list, discarded_sentences_list)

            # build a sentences
            lxr.build_summarization_html()

    # log results after finish algorithm
    def log_results(self, summary, sorted_ix, lex_scores, description_summary_list, discarded_sentences_list):
        '''
        log results of LexRank algorithm
        '''
        summary_below_threshold = list()
        sorted_ix_below_threshold = list()

        Logger.info('')
        Logger.info('summary extracted:')
        for sen_idx, sentence in enumerate(summary):
            Logger.info('idx: {}, score: {} - {}'.format(
                str(sorted_ix[sen_idx]), str(round(lex_scores[sorted_ix[sen_idx]], 3)), str(sentence.encode('utf-8'))
            ))
            if sorted_ix[sen_idx] not in discarded_sentences_list:
                summary_below_threshold.append(sentence.encode('utf-8'))
                sorted_ix_below_threshold.append(sorted_ix[sen_idx])

        Logger.info('')
        Logger.info('sentence order:')
        Logger.info(sorted_ix)
        Logger.info('')
        Logger.info('sentence rank:')
        Logger.info(lex_scores)
        Logger.info('')
        Logger.info('Summary output')
        Logger.info(description_summary_list)

        # self.log_html_format(summary, sorted_ix, discarded_sentences_list)
        self.log_html_format(summary_below_threshold, sorted_ix_below_threshold)
        # self.log_html_format(summary_below_threshold, sorted_ix_below_threshold)

    @staticmethod
    def log_html_format(summary, sorted_ix):
        """
        :return: summary in HTML format, in length K (HTML_SUMMARY_SIZE) and ordered properly
        """

        relevant_summary = summary[:HTML_SUMMARY_SIZE]  # remain only first K sentences
        relevant_sorted_ix = sorted_ix[:HTML_SUMMARY_SIZE]  # remain only first K sentences original place

        def sort_list(list1, list2):
            zipped_pairs = zip(list2, list1)
            z = [x for _, x in sorted(zipped_pairs)]
            return z

        ordered_summary = sort_list(relevant_summary, relevant_sorted_ix)
        ordered_summary = [sen[0].upper() + sen[1:] for sen in ordered_summary]
        Logger.info('')
        Logger.info(LEX_RANK_ALGORITHM_VERSION)
        Logger.info(PERSONALITY_TRAIT_DICT)
        Logger.info('')
        Logger.info('Summary in HTML format')
        Logger.info(". <br/>".join(ordered_summary) + ".")
        Logger.info('')


def main():
    LexRankObj = WrapperLexRank()
    LexRankObj.init_debug_log()
    LexRankObj.test_lexrank()


if __name__ == '__main__':
    main()


