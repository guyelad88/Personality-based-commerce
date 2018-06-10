import logging
import pandas as pd

from summarizer import LexRank
from calculate_word_contribute import CalculateWordContribute


# Run personality-based LexRank algorithm.
class WrapperLexRank:

    def __init__(self, corpus_path_file, log_dir, html_dir, target_item_description_file, trait_word_contribute_folder,
                 personality_trait_dict, lex_rank_algorithm_version, summarization_version, personality_word_flag,
                 random_walk_flag, damping_factor, summarization_similarity_threshold, target_sentences_length,
                 summary_size, threshold, multi_document_summarization, corpus_size):

        self.corpus_path_file = corpus_path_file
        self.log_dir = log_dir
        self.html_dir = html_dir
        self.target_item_description_file = target_item_description_file
        self.trait_word_contribute_folder = trait_word_contribute_folder
        self.personality_trait_dict = personality_trait_dict
        self.lex_rank_algorithm_version = lex_rank_algorithm_version    # LexRank version
        self.summarization_version = summarization_version              # summarization version
        self.personality_word_flag = personality_word_flag          # flag - calculate similarity adapt to personality
        self.random_walk_flag = random_walk_flag                    # bool if to use random jump
        self.damping_factor = damping_factor                        # probability to jump random factor
        self.summarization_similarity_threshold = summarization_similarity_threshold    # max similarity in top-relevant
        self.summary_size = summary_size
        self.target_sentences_length = target_sentences_length
        self.threshold = threshold
        self.multi_document_summarization = multi_document_summarization  # summarization multi/single method
        self.corpus_size = corpus_size      # corpus size to calculate idf from

        self.verbose_flag = True

        an = self
        self.input_attr_dict = vars(an)         # save attribute to show them later

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.word_cont_dict = dict()    # word and correspond contribute value
        return

    # build log object
    def init_debug_log(self):
        import logging

        lod_file_name = self.log_dir + 'LexRank_algorithm_' + str(self.cur_time) + '.log'

        # logging.getLogger().addHandler(logging.StreamHandler())

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

    # extract corpus documents from a file
    def load_corpus_documents(self, corpus_file):

        '''
        :param corpus_file: path to file contain corpuse data (documents)
        :return: list of string, each str contain item description
        '''

        description_df = pd.read_csv(corpus_file)  # items and their descriptions
        description_df = description_df['description'].values.astype(str)     # description_df[['item_id', 'description']]
        documents = description_df.tolist()
        return documents

    # load item description to summarize
    def load_description_df(self):

        target_description_df = pd.read_csv(self.target_item_description_file)  # items and their descriptions

        return target_description_df

    # check arguments are valid
    def check_input(self):

        for c_trait, user_value in self.personality_trait_dict.iteritems():
            if user_value not in ['H', 'L', 'M']:
                raise('unknown trait value')

        if self.damping_factor < 0 or self.damping_factor > 1:
            raise('damping factor must be a float between 0 to 1')

        if self.summarization_similarity_threshold < 0 or self.summarization_similarity_threshold > 1:
            raise('summarization similarity threshold must be a float between 0 to 1')

        if 'min' not in self.target_sentences_length or 'max' not in self.target_sentences_length:
            raise ('target_sentences_length dictionary must contain min and max keys')

        if self.lex_rank_algorithm_version not in ['vanilla-LexRank', 'personality-based-LexRank']:
            raise ('unknown lex_rank_algorithm_version')

        if self.summarization_version not in ['top_relevant', 'Bollegata', 'Shahaf']:
            raise ('unknown summarization_version')

        if self.multi_document_summarization not in ['single', 'multi']:
            raise ('unknown multi_document_summarization value (not single nor multi)')

        if self.corpus_size != 'max' and not isinstance(self.corpus_size, int):
            raise('unknown corpus size variable')

        return

    # log class arguments
    def log_attribute_input(self):

        logging.info('')
        logging.info('Class arguments')
        for attr, attr_value in self.input_attr_dict.iteritems():
            logging.info('Attribute: ' + str(attr) + ', Value: ' + str(attr_value))
        logging.info('')
        return

    # main function - run LexRank
    def test_lexrank(self):
        '''
        :return: run LexRank class with corpus, list of sentences to summary and configuration
        '''

        self.check_input()              # check inputs are valid
        self.log_attribute_input()      # log class arguments

        # calculate word contribute to current personality
        word_contibute_obj = CalculateWordContribute(self.trait_word_contribute_folder, self.personality_trait_dict, logging)
        word_contibute_obj.calculate_user_total_word_contribute()
        word_cont_dict = word_contibute_obj.meta_word_contribute        # personality word contribute

        # collect corpus data
        documents = self.load_corpus_documents(self.corpus_path_file)

        # load item description to summarize
        target_description_df = self.load_description_df()

        for index, cur_row_target_description in target_description_df.iterrows():

            desc_id = cur_row_target_description['ID']   # ID,TITLE,DESCRIPTION
            desc_title = cur_row_target_description['TITLE']  # ID,TITLE,DESCRIPTION

            # desc_desc = cur_row_target_description['DESCRIPTION']  # ID,TITLE,DESCRIPTION

            desc_1 = cur_row_target_description['DESCRIPTION_1']  # ID,TITLE,DESCRIPTION
            desc_2 = cur_row_target_description['DESCRIPTION_2']  # ID,TITLE,DESCRIPTION
            if pd.isnull(desc_1) or pd.isnull(desc_2):
                continue

            desc_desc = desc_1 + '. ' + desc_2

            if desc_id == 'B002C30S96':
                a = 5
            else:
                continue
            target_sentences = self.clean_target_sentences(desc_desc)

            # limitation for number of sentences to summarize
            if len(target_sentences) > self.target_sentences_length['max'] \
                    or len(target_sentences) < self.target_sentences_length['min']:     # control amount of sentences
                continue

            # if desc_id != 'B00J00X5YO':
            #    continue

            # define summarization length
            if self.summary_size == 'max':
                summary_size = len(target_sentences)
            else:
                summary_size = self.summary_size

            logging.info('')
            logging.info('Description ID: ' + str(desc_id) + ', Length: ' + str(len(desc_desc)) + ', Title: ' +
                         str(desc_title))
            logging.info('item sentences: ' + str(len(target_sentences)))

            # build LexRank class obj, calculate idf on entire corpus
            lxr = LexRank(
                documents,
                logging,
                lex_rank_algorithm_version,
                summarization_version,
                html_dir=html_dir,
                user_personality_dict=self.personality_trait_dict,
                word_cont_dict=word_cont_dict,
                personality_word_flag=personality_word_flag,        # similarity based personality
                random_walk_flag=self.random_walk_flag,
                damping_factor=self.damping_factor,
                summarization_similarity_threshold=self.summarization_similarity_threshold,
                multi_document_summarization=self.multi_document_summarization,   # multi/single summarization
                cur_time=self.cur_time,
                desc_title=desc_title,
                desc_id=desc_id,
                corpus_size=self.corpus_size,
                stopwords=None,                                     # STOPWORDS['en'],
                keep_numbers=False,
                keep_emails=False,
                include_new_words=True,
            )

            # build similarity matrix, power method
            summary, sorted_ix, lex_scores, description_summary_list = lxr.get_summary(
                target_sentences,
                summary_size,
                self.threshold,
                discretize=False)

            # write results into log file
            self.log_results(summary, sorted_ix, lex_scores, description_summary_list)

            # build a sentences
            lxr.build_summarization_html()

    # use only sentence with at least 5 char
    def clean_target_sentences(self, description_str):

        list_valid_sentences = description_str.split('.')

        # remove sentences with less than 5 words
        list_valid_sentences = [x for x in list_valid_sentences if len(x.split(' ')) > 6]

        clean_list = list()
        for sen in list_valid_sentences:
            if sen[0] == ' ':
                clean_list.append(sen[1:])
            else:
                clean_list.append(sen)
        # remove duplication
        final_clean_list = list()
        for i in clean_list:
            if i not in final_clean_list:
                final_clean_list.append(i)
        return final_clean_list

    # log results after finish algorithm
    def log_results(self, summary, sorted_ix, lex_scores, description_summary_list):
        '''
        log results of LexRank algorithm
        '''
        logging.info('')
        logging.info('summary extracted:')
        for sen_idx, sentence in enumerate(summary):
            logging.info('idx: ' + str(sorted_ix[sen_idx]) + ', score: ' + str(round(lex_scores[sorted_ix[sen_idx]], 3))
                         + ' - ' + str(sentence))
        logging.info('')
        logging.info('sentence order:')
        logging.info(sorted_ix)
        logging.info('')
        logging.info('sentence rank:')
        logging.info(lex_scores)
        logging.info('')
        logging.info('Summary output')
        logging.info(description_summary_list)

        return


def main(corpus_path_file, log_dir, html_dir, target_item_description_file, trait_word_contribute_folder, personality_trait_dict,
         lex_rank_algorithm_version, summarization_version, personality_word_flag, random_walk_flag, damping_factor,
         summarization_similarity_threshold, target_sentences_length, summary_size, threshold, multi_document_summarization, corpus_size):

    LexRankObj = WrapperLexRank(corpus_path_file, log_dir, html_dir, target_item_description_file, trait_word_contribute_folder,
                                personality_trait_dict, lex_rank_algorithm_version, summarization_version,
                                personality_word_flag, random_walk_flag, damping_factor,
                                summarization_similarity_threshold, target_sentences_length, summary_size, threshold, multi_document_summarization, corpus_size)
    LexRankObj.init_debug_log()         # init log file
    LexRankObj.test_lexrank()           #


if __name__ == '__main__':
    # run LexRank algorithm
    # TODO change to dir path
    corpus_path_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/predict_personality_from_descriptions/descriptions/num_items_1552_2018-01-30 13:15:33.csv'
    log_dir = 'log/'
    html_dir = 'html/'
    summary_size = 4 # 'max'         # summarization length - 'max' or int
    threshold = 0.03             # min edge weight between two sentences
    damping_factor = 0.01        # probability not to jump
    summarization_similarity_threshold = 0.3
    target_sentences_length = {
        'min': 20,
        'max': 23
    }
    corpus_size = 'max'
    # please don't change (damping factor=1 same effect)
    personality_word_flag = True
    random_walk_flag = True                         # flag if combine random jump between sentences
    # TODO infer from data
    multi_document_summarization = 'single'         # summarization method for single/multi documents

    lex_rank_algorithm_version = 'personality-based-LexRank'   # 'vanilla-LexRank', 'personality-based-LexRank'
    summarization_version = 'top_relevant'          # 'top_relevant', 'Bollegata', 'Shahaf'

    # current user personality - 'H'\'L'\'M' (high\low\miss(or mean))
    personality_trait_dict = {
        'openness': 'H',
        # 'conscientiousness': 'L',
        'extraversion': 'H',
        'agreeableness': 'H',
        'neuroticism': 'H'
    }

    trait_word_contribute_folder = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/kl/results/all_words_contribute'

    # TODO
    # TODO add contribute to graph weight regards to traits
    # TODO personalization - binary per trait - load relevant file

    target_item_description_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/amazon_description/amazon_1881.csv'
    main(corpus_path_file, log_dir, html_dir, target_item_description_file, trait_word_contribute_folder,
         personality_trait_dict, lex_rank_algorithm_version, summarization_version, personality_word_flag,
         random_walk_flag, damping_factor, summarization_similarity_threshold, target_sentences_length, summary_size,
         threshold, multi_document_summarization, corpus_size)


