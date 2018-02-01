import sys
import csv
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# input: two descriptions file represent all texts of specific traits/vertical
# output: KL divergence of language model + contribute words for KL metric
class CreateVocabularies:

    def __init__(self, description_file_p, description_file_q, log_dir, results_dir, vocabulary_method,
                 results_dir_title, verbose_flag):

        # arguments
        self.description_file_p = description_file_p    # description file
        self.description_file_q = description_file_q    # description file
        self.log_dir = log_dir                          # log directory
        self.results_dir = results_dir                  # result directory
        self.vocabulary_method = vocabulary_method      # cal klPost, klCalc
        self.results_dir_title = results_dir_title      # init name to results file
        self.verbose_flag = verbose_flag                # print results in addition to log file

        import ntpath
        self.file_name_p = ntpath.basename(self.description_file_p)[:-4].split('_')[1]
        self.file_name_q = ntpath.basename(self.description_file_q)[:-4].split('_')[1]

        self.top_k_words = 30           # present top words
        self.SMOOTHING_FACTOR = 1.0     # smoothing factor for calculate term contribution

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.corpus = list()        # corpus contain two languagas models
        self.vectorizer = list()    # transform words into vectors of numbers using sklearn
        self.X = list()             # sparse vector represent vocabulary words
        self.X_dense = list()       # dense vector represent vocabulary words

        self.X_dense_p = list()     # dense tf for p distribution
        self.X_dense_q = list()     # dense tf for q distribution
        self.X_dense_kl = list()    # matrix contain two dense vectors for kl

        # documents method
        self.count_vec_p = CountVectorizer()
        self.count_vec_q = CountVectorizer()
        self.occurrence_doc_sum_p = list()
        self.occurrence_doc_sum_q = list()
        self.len_p = int                        # number of posts
        self.len_q = int                        # number of posts

        self.normalize_q = list()   # normalize
        self.normalize_p = list()
        self.cur_time = str
        csv.field_size_limit(sys.maxsize)

    # build log object
    def init_debug_log(self):
        import logging
        from time import gmtime, strftime

        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        lod_file_name = self.log_dir + 'calculate_kl_' + str(self.cur_time) + '.log'

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

        logging.info("start log program")

    def check_input(self):

        if self.vocabulary_method not in ['documents', 'aggregation']:
            raise ('vocabulary method: ' + str(self.vocabulary_method) + ' is not defined')

        return

    # load vocabularies regards to vocabulary method
    def run_kl(self):

        if self.vocabulary_method == 'documents':
            self.load_vocabulary_vector_documents()
            self.calculate_kl_and_language_models_documents()
        elif self.vocabulary_method == 'aggregation':
            self.load_vocabulary_vector_aggregation()
            self.calculate_kl_and_language_models_aggregation()
        return

    # load vocabulary support aggregation method - KLPost
    def load_vocabulary_vector_documents(self):

        import pickle
        import scipy
        import numpy

        with open(self.description_file_p, 'rb') as fp:
            text_list_list_p = pickle.load(fp)
            self.len_p = np.float(len(text_list_list_p))
            logging.info('P #of items descriptions: ' + str(self.len_p))

        with open(self.description_file_q, 'rb') as fp:
            text_list_list_q = pickle.load(fp)
            self.len_q = np.float(len(text_list_list_q))
            logging.info('Q #of items descriptions: ' + str(self.len_q))

        #  P distribution
        self.count_vec_p = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        X_train_counts_p = self.count_vec_p.fit_transform(text_list_list_p)   # count occurrence per word in documents
        X_dense_p = scipy.sparse.csr_matrix.todense(X_train_counts_p)   # dense transformation
        X_dense_binary_p = scipy.sign(X_dense_p)                        # binary - count 1-0 occurrence per documents
        self.occurrence_doc_sum_p = numpy.sum(X_dense_binary_p, axis=0).transpose()     # sum occurrence oer documents
        self.occurrence_doc_sum_p = self.occurrence_doc_sum_p.tolist()

        #  Q distribution
        self.count_vec_q = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        X_train_counts_q = self.count_vec_q.fit_transform(text_list_list_q)  # count occurrence per word in documents
        X_dense_q = scipy.sparse.csr_matrix.todense(X_train_counts_q)  # dense transformation
        X_dense_binary_q = scipy.sign(X_dense_q)  # binary - count 1-0 occurrence per documents
        self.occurrence_doc_sum_q = numpy.sum(X_dense_binary_q, axis=0).transpose()  # sum occurrence oer documents
        self.occurrence_doc_sum_q = self.occurrence_doc_sum_q.tolist()

        return

    # load vocabulary support aggregation method - KLCalc, using sklearn
    def load_vocabulary_vector_aggregation(self):
        text_file_p = open(self.description_file_p, "r")
        text_str_p = text_file_p.read().replace('\n', ' ')

        text_file_q = open(self.description_file_q, "r")
        text_str_q = text_file_q.read().replace('\n', ' ')

        logging.info('P #of words: ' + str(len(text_str_p.split())))
        logging.info('Q #of words: ' + str(len(text_str_q.split())))

        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )

        self.corpus = [
            text_str_p,
            text_str_q,
        ]
        self.X = self.vectorizer.fit_transform(self.corpus)
        return

    # calculate KL values
    def calculate_kl(self):

        import scipy

        self.X_dense = scipy.sparse.csr_matrix.todense(self.X)

        import numpy as np
        self.X_dense = self.X_dense.astype(np.float)

        import copy
        self.X_dense_kl = copy.deepcopy(self.X_dense)

        # smooth zeros
        for (x, y), value in np.ndenumerate(self.X_dense):
            if value == 0:
                self.X_dense_kl[x, y] = 0.01

        # normalize vectors
        sum_of_p = sum(self.X_dense_kl[0].tolist()[0])
        self.normalize_p = [x / sum_of_p for x in self.X_dense_kl[0].tolist()[0]]
        sum_of_q = sum(self.X_dense_kl[1].tolist()[0])
        self.normalize_q = [x / sum_of_q for x in self.X_dense_kl[1].tolist()[0]]

        # calculate KL
        kl_1 = scipy.stats.entropy(self.normalize_p, self.normalize_q)
        kl_2 = scipy.stats.entropy(self.normalize_q, self.normalize_p)

        logging.info('KL value 1: ' + str(kl_1))
        logging.info('KL value 2: ' + str(kl_2))
        return

    # calculate KL results (both), and most separate values
    def calculate_kl_and_language_models_documents(self):

        # TODO self.calculate_kl()  # cal kl using sc ipy

        # cal most significant separate words - both direction
        name = 'P_' + str(self.file_name_p) + '_Q_' + str(self.file_name_q)
        logging.info(name)
        self.calculate_separate_words_documents(self.occurrence_doc_sum_p, self.occurrence_doc_sum_q,
                                                self.count_vec_p.vocabulary_, self.count_vec_q.vocabulary_,
                                                self.len_p, self.len_q, name)
        logging.info('')
        name = 'P_' + str(self.file_name_q) + '_Q_' + str(self.file_name_p)
        logging.info(name)
        self.calculate_separate_words_documents(self.occurrence_doc_sum_q, self.occurrence_doc_sum_p,
                                                self.count_vec_q.vocabulary_, self.count_vec_p.vocabulary_,
                                                self.len_q, self.len_p, name)
        return

    # calculate KL results (both), and most separate values
    def calculate_kl_and_language_models_aggregation(self):

        self.calculate_kl()  # cal kl using sc ipy

        # cal most significant separate words - both direction
        self.calculate_separate_words_aggregation(self.X_dense[0].tolist()[0], self.X_dense[1].tolist()[0])
        self.calculate_separate_words_aggregation(self.X_dense[1].tolist()[0], self.X_dense[0].tolist()[0])
        return

    # calculate most significance term to KL divergence
    def calculate_separate_words_documents(self, X_p, X_q, dict_p, dict_q, len_p, len_q, name):

        dict_ratio = dict()  # contain word index and his KL contribute
        inv_p = {v: k for k, v in dict_p.iteritems()}
        inv_q = {v: k for k, v in dict_q.iteritems()}

        for word_idx_p, tf_p in enumerate(X_p):

            tf_p = np.float(tf_p[0]/len_p)
            # word_p = dict_p.keys()[dict_p.values().index(word_idx_p)]
            word_p = inv_p[word_idx_p]

            if word_p not in dict_q:
                tf_q = 1.0 / (len_q * self.SMOOTHING_FACTOR)        # smooth q
            else:
                tf_q = np.float(X_q[dict_q[word_p]][0]/len_q)

            contribute = tf_p * np.log(tf_p / tf_q)
            dict_ratio[word_idx_p] = contribute

        # find top k
        import xlwt
        import operator
        dict_max_ratio = sorted(dict_ratio.items(), key=operator.itemgetter(1))
        dict_max_ratio.reverse()
        counter = 0

        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet(name)
        sheet1.write(0, 0, 'Word')
        sheet1.write(0, 1, 'TF_P')
        sheet1.write(0, 2, 'TF_Q')
        sheet1.write(0, 3, 'Contribute')
        sheet1.write(0, 4, 'Fraction_P')
        sheet1.write(0, 5, 'Fraction_Q')

        for idx, tup in enumerate(dict_max_ratio):

            counter += 1
            if counter > self.top_k_words:
                break

            cur_word = dict_p.keys()[dict_p.values().index(tup[0])]
            tf_p = X_p[tup[0]][0]
            if cur_word not in dict_q:
                tf_q = 0
            else:
                tf_q = X_q[dict_q[cur_word]][0]

            frac_p = np.float(tf_p) / len_p
            frac_q = np.float(tf_q) / len_q
            logging.info(str(cur_word) + ', tf p: ' + str(tf_p) + ', tf q: ' + str(tf_q) + ', cont: ' +
                         str(round(tup[1], 2)) + ', ratio_tf p: ' + str(round(frac_p, 3)) +
                         ', ratio_tf q: ' + str(round(frac_q, 3)))
            sheet1.write(idx+1, 0, str(cur_word))
            sheet1.write(idx+1, 1, str(tf_p))
            sheet1.write(idx+1, 2, str(tf_q))
            sheet1.write(idx+1, 3, str(round(tup[1], 2)))
            sheet1.write(idx+1, 4, str(round(frac_p, 3)))
            sheet1.write(idx+1, 5, str(round(frac_q, 3)))

        excel_file_name = self.results_dir + str(results_dir_title) + 'K_' + str(self.top_k_words) + '_Smooth_' + \
                          str(self.SMOOTHING_FACTOR) + '_' +str(self.vocabulary_method) + '_' + str(name) + \
                          '_top_' + str(self.top_k_words) + '_' + str(self.cur_time) + '.xls'

        book.save(excel_file_name)
        logging.info('save kl in file: ' + str(excel_file_name))
        logging.info('')
        logging.info('')
        return

    # calculate most significance term to KL divergence
    def calculate_separate_words_aggregation(self, X_p, X_q):

        self.X_dense_p = X_p    # self.X_dense[0].tolist()[0]
        self.X_dense_q = X_q    # self.X_dense[1].tolist()[0]

        sum_of_p = np.float(sum(self.X_dense_p))
        sum_of_q = np.float(sum(self.X_dense_q))

        dict_ratio = dict()     # contain word index and his KL contribute

        for word_idx, tf_p in enumerate(self.X_dense_p):

            if tf_p > 0:
                tf_p = np.float(tf_p)/sum_of_p
                tf_q = np.float(self.X_dense_q[word_idx])/sum_of_q

                # smoothing
                if tf_q == 0:
                    tf_q = 1.0 / (sum_of_q * self.SMOOTHING_FACTOR)

                contribute = tf_p * np.log(tf_p/tf_q)
                dict_ratio[word_idx] = contribute

        # find top k
        import operator
        dict_max_ratio = sorted(dict_ratio.items(), key=operator.itemgetter(1))
        dict_max_ratio.reverse()
        counter = 0

        for idx, tup in enumerate(dict_max_ratio):

            counter += 1
            if counter > self.top_k_words:
                break

            tf_p = self.X_dense_p[tup[0]]
            tf_q = self.X_dense_q[tup[0]]
            frac_p = np.float(tf_p) / sum_of_p
            frac_q = np.float(tf_q) / sum_of_q
            cur_ratio = self.vectorizer.vocabulary_.keys()[self.vectorizer.vocabulary_.values().index(tup[0])]
            logging.info(str(cur_ratio) + ', tf p: ' + str(tf_p) + ', tf q: ' + str(tf_q) + ', cont: ' +
                         str(round(tup[1], 2)) + ', ratio_tf p: ' + str(round(frac_p, 3)) +
                         ', ratio_tf q: ' + str(round(frac_q, 3)))

        logging.info('')
        logging.info('')

        return

    # calculate term with largest ratio and present them
    def cal_ratio_tf_q_p(self, dict_max_ratio):
        counter = 0
        for idx, tup in enumerate(dict_max_ratio):
            counter += 1
            if counter > self.top_k_words:
                break
            tf_p = self.X_dense.tolist()[0][tup[0]]
            tf_q = self.X_dense.tolist()[1][tup[0]]
            cur_ratio = self.vectorizer.vocabulary_.keys()[self.vectorizer.vocabulary_.values().index(tup[0])]
            logging.info(str(cur_ratio) + ', tf p: ' + str(tf_p) + ', tf q: ' + str(tf_q))
        return


def main(description_file_p, description_file_q, log_dir, results_dir, vocabulary_method, results_dir_title,
         verbose_flag):

    # init class
    create_vocabularies_obj = CreateVocabularies(description_file_p, description_file_q, log_dir, results_dir,
                                                 vocabulary_method, results_dir_title, verbose_flag)

    create_vocabularies_obj.init_debug_log()                    # init log file
    create_vocabularies_obj.check_input()                       # check if arguments are valid
    create_vocabularies_obj.run_kl()                            # contain all inner functions


if __name__ == '__main__':

    # item and hist description file
    # description_file_p = '/Users/sguyelad/PycharmProjects/research/kl/vocabulary/2018-01-30 11:15:54/documents_Home & Garden.txt'
    # description_file_q = '/Users/sguyelad/PycharmProjects/research/kl/vocabulary/2018-01-30 11:15:54/documents_Collectibles.txt'

    # extraversion
    # description_file_p = '../vocabulary/2018-02-01 13:16:22/documents_high_extraversion.txt'
    # description_file_q = '../vocabulary/2018-02-01 13:16:22/documents_low_extraversion.txt'

    # openness
    # description_file_p = '../vocabulary/2018-02-01 13:18:02/documents_high_openness.txt'
    # description_file_q = '../vocabulary/2018-02-01 13:18:02/documents_low_openness.txt'

    # agreeableness
    # description_file_p = '../vocabulary/2018-02-01 13:14:08/documents_high_agreeableness.txt'
    # description_file_q = '../vocabulary/2018-02-01 13:14:08/documents_low_agreeableness.txt'

    # conscientiousness
    description_file_p = 'vocabulary/2018-02-01 13:42:52/documents_high_conscientiousness.txt'
    description_file_q = 'vocabulary/2018-02-01 13:42:52/documents_low_conscientiousness.txt'

    # neuroticism
    # description_file_p = '../vocabulary/2018-02-01 12:55:36/documents_high_neuroticism.txt'
    # description_file_q = '../vocabulary/2018-02-01 12:55:36/documents_low_neuroticism.txt'

    log_dir = 'log/'
    results_dir = 'results/'
    results_dir_title = 'conscientiousness_05_gap_'
    verbose_flag = True
    vocabulary_method = 'documents'     # 'documents', 'aggregation'

    main(description_file_p, description_file_q, log_dir, results_dir, vocabulary_method, results_dir_title,
         verbose_flag)
