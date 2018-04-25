from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn
import scipy
from classifier import PredictDescriptionModel


class WrapperModel:
    '''
    run embedding+LSVM / tf-ifd+RR classifiers using differents configuration
    '''

    def __init__(self, file_directory, log_dir, eli_5_dir, plot_dir, load_already_split_df_bool, model_type, equal_labels=True):

        # file arguments
        self.file_directory = file_directory    # directory contain all data for all traits
        self.log_dir = log_dir
        self.equal_labels = equal_labels        # number of items in each group will be equal - min(i1,i2)
        self.verbose_flag = True
        self.eli_5_dir = eli_5_dir
        self.plot_dir = plot_dir
        self.model_type = model_type            # 'n-gram-rr' / 'lstm'

        # bool - load split df / split inside classifier class
        self.load_already_split_df_bool = load_already_split_df_bool

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # tf-idf + RR
        self.estimator_type = ['NuSVC', 'AdaBoost', 'logistic_cv', 'logistic']
        self.models_types = ['tf-idf', 'n-gram']
        self.ngram_range = [(1, 1), (1, 2), (1, 3)]
        self.norm = ['l1', 'l2']
        self.max_df = [0.25, 0.5, 0.75, 1.0]
        self.min_df = [1, 5, 10]

        # lstm
        self.embedding_size_list = list()
        self.num_epoch_list = list()
        self.maxlen_list = list()
        self.batch_size_list = list()
        self.dropout_list = list()
        self.max_num_words = list()

        self.num_model_total_to_check = len(self.estimator_type) * len(self.models_types) * len(self.ngram_range) * \
                                        len(self.norm) * len(self.max_df) * len(self.min_df)

        self.use_idf = True
        self.smooth_idf = True
        self.sublinear_tf = False

        # self.min_df = 1         # int - minimum number of doc, float - min prop of docs
        # self.max_df = 0.75      # float - max prop of doc he appears in

        # dictionary to contain score and find at the end the best scores
        self.openness_accuracy = dict()
        self.conscientiousness_accuracy = dict()
        self.extraversion_accuracy = dict()
        self.agreeableness_accuracy = dict()
        self.neuroticism_accuracy = dict()

        self.openness_acu = dict()
        self.conscientiousness_acu = dict()
        self.extraversion_acu = dict()
        self.agreeableness_acu = dict()
        self.neuroticism_acu = dict()

        # object contain object to sort for extracting best configuration
        self.trait_accuracy_dict = {
            'openness': self.openness_accuracy,
            'conscientiousness': self.conscientiousness_accuracy,
            'extraversion': self.extraversion_accuracy,
            'agreeableness': self.agreeableness_accuracy,
            'neuroticism': self.neuroticism_accuracy,
        }

        self.trait_auc_dict = {
            'openness': self.openness_acu,
            'conscientiousness': self.conscientiousness_acu,
            'extraversion': self.extraversion_acu,
            'agreeableness': self.agreeableness_acu,
            'neuroticism': self.neuroticism_acu,
        }

        self.num_models = 0

    # build log object
    def init_debug_log(self):
        import logging

        lod_file_name = self.log_dir + 'predict_personality_from_desc_' + str(self.cur_time) + '.log'

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
        logging.info("start log program")
        return

    # run models
    def run_experiments(self):

        if self.model_type == 'n-gram-rr':
            self.run_experiments_n_gram_rr()
        elif self.model_type == 'lstm':
            self.run_experiments_lstm()
        else:
            raise('unknown model type: ' + str(self.model_type))
        return

    #
    def run_experiments_n_gram_rr(self):

        for cur_estimator in self.estimator_type:   # logistic/logistic_cv/AdaBoost/NuSVC
            for cur_model in self.models_types:         # tf-idf, n-gram
                for cur_ngram in self.ngram_range:      # n-gram
                    for cur_norm in self.norm:          # norm used
                        for cur_max_df in self.max_df:
                            for cur_min_df in self.min_df:
                                self.num_models += 1
                                logging.info('model number: ' + str(self.num_models) + ' / ' +
                                             str(self.num_model_total_to_check) + ' - ' +
                                             str(round(float(self.num_models)/float(self.num_model_total_to_check), 2))
                                             + '%')
                                logging.info('Est: ' + str(cur_estimator) + ', Model: ' + str(cur_model) + ', N-Gram: ' + str(cur_ngram) + ', Norm: ' + str(cur_norm))#  + ', max_df: ' + str(cur_max_df) + ', min_df: ' + str(cur_min_df)
                                cur_key = 'est_' + str(cur_estimator) + '_model_' + str(cur_model) + '_ngram_' + str(cur_ngram) + '_norm_' + str(cur_norm) + '_max_df_' + str(cur_max_df) + '_min_df_' + str(cur_min_df)
                                logging.info('Current key: ' + str(cur_key))
                                predict_desc_obj = PredictDescriptionModel(file_directory,
                                                                           log_dir,
                                                                           self.eli_5_dir,
                                                                           self.plot_dir,
                                                                           self.load_already_split_df_bool,
                                                                           self.model_type,     # 'n-gram-rr'
                                                                           cur_estimator,
                                                                           cur_model,
                                                                           cur_ngram,
                                                                           cur_norm,
                                                                           self.use_idf,
                                                                           self.smooth_idf,
                                                                           self.sublinear_tf,
                                                                           cur_min_df,
                                                                           cur_max_df,
                                                                           cur_key,
                                                                           self.cur_time,
                                                                           equal_labels=True)
                                # predict_desc_obj.init_debug_log()       # init log file
                                predict_desc_obj.run_experiment()
                                self.insert_scores_to_dict(predict_desc_obj, cur_key)
        # sort and find best configuration for each personality trait
        self.summarize_results()

        return

    # lstm -
    def run_experiments_lstm(self):

        '''logging.info('Est: ' + str(cur_estimator) + ', Model: ' + str(cur_model) + ', N-Gram: ' + str(
            cur_ngram) + ', Norm: ' + str(
            cur_norm))  # + ', max_df: ' + str(cur_max_df) + ', min_df: ' + str(cur_min_df) '''

        self.embedding_size_list = [128, 256]
        self.num_epoch_list = [30]      # , 30]                   # [10, 20, 30]
        self.maxlen_list = [200, 300]        # [100, 200, 300]           #, 300]
        self.batch_size_list = [16, 32]
        self.dropout_list = [0.2, 0.3]
        self.max_num_words = [None]  # , 10000]

        for embedding_size in self.embedding_size_list:
            for num_epoch in self.num_epoch_list:
                for maxlen in self.maxlen_list:
                    for batch_size in self.batch_size_list:
                        for dropout in self.dropout_list:
                            for max_num_words in self.max_num_words:
                                lstm_parameters_dict = {
                                    'max_features': 200000,
                                    'maxlen': maxlen,
                                    'batch_size': batch_size,
                                    'embedding_size': embedding_size,
                                    'num_epoch': num_epoch,
                                    'dropout': dropout,     # 0.2
                                    'recurrent_dropout': dropout,   # 0.2
                                    'tensor_board_bool': False,
                                    'max_num_words': max_num_words
                                }

                                logging.info('start LSTM model')

                                # cur_key = 'est_' + str(cur_estimator) + '_model_' + str(cur_model) + '_ngram_' + str(
                                #     cur_ngram) + '_norm_' + str(cur_norm) + '_max_df_' + str(cur_max_df) + '_min_df_' + str(cur_min_df)

                                cur_key = 'max_len_' + str(maxlen) + '_batch_size_' + str(batch_size) + '_embedding_size_' + \
                                          str(embedding_size) + '_num_epoch_' + str(num_epoch) + '_dropout_' + str(dropout) + '_max_num_words_' + str(max_num_words)

                                logging.info('Current key: ' + str(cur_key))

                                predict_desc_obj = PredictDescriptionModel(file_directory,
                                                                           log_dir,
                                                                           self.eli_5_dir,
                                                                           self.plot_dir,
                                                                           self.load_already_split_df_bool,
                                                                           self.model_type,     # 'lstm'
                                                                           '',  # cur_estimator,
                                                                           '',  # cur_model,
                                                                           '',  # cur_ngram,
                                                                           '',  # cur_norm,
                                                                           self.use_idf,
                                                                           self.smooth_idf,
                                                                           self.sublinear_tf,
                                                                           '',  # cur_min_df,
                                                                           '',  # cur_max_df,
                                                                           cur_key,
                                                                           self.cur_time,
                                                                           lstm_parameters_dict,
                                                                           equal_labels=True)
                                # predict_desc_obj.init_debug_log()       # init log file
                                predict_desc_obj.run_experiment()
                                self.insert_scores_to_dict(predict_desc_obj, cur_key)
                # sort and find best configuration for each personality trait

                                self.summarize_results()

        self.summarize_results()

        return

    # insert score for last configuration
    def insert_scores_to_dict(self, cur_obj, cur_key):

        try:
            self.openness_accuracy[cur_key] = cur_obj.predict_personality_accuracy['openness']
            self.conscientiousness_accuracy[cur_key] = cur_obj.predict_personality_accuracy['conscientiousness']
            self.extraversion_accuracy[cur_key] = cur_obj.predict_personality_accuracy['extraversion']
            self.agreeableness_accuracy[cur_key] = cur_obj.predict_personality_accuracy['agreeableness']
            self.neuroticism_accuracy[cur_key] = cur_obj.predict_personality_accuracy['neuroticism']
        except:
            logging.info('Attr missing: ' + 'predict_personality_accuracy')
            pass

        try:
            self.openness_acu[cur_key] = cur_obj.predict_personality_AUC['openness']
            self.conscientiousness_acu[cur_key] = cur_obj.predict_personality_AUC['conscientiousness']
            self.extraversion_acu[cur_key] = cur_obj.predict_personality_AUC['extraversion']
            self.agreeableness_acu[cur_key] = cur_obj.predict_personality_AUC['agreeableness']
            self.neuroticism_acu[cur_key] = cur_obj.predict_personality_AUC['neuroticism']
        except:
            logging.info('Attr missing: ' + 'predict_personality_AUC')
            pass

        return

    # init score dictionary, which finally will contain all model scores
    def init_score_dict(self):

        self.openness_accuracy = dict()
        self.conscientiousness_accuracy = dict()
        self.extraversion_accuracy = dict()
        self.agreeableness_accuracy = dict()
        self.neuroticism_accuracy = dict()

        self.openness_acu = dict()
        self.conscientiousness_acu = dict()
        self.extraversion_acu = dict()
        self.agreeableness_acu = dict()
        self.neuroticism_acu = dict()

        return

    # sort and present best models
    def summarize_results(self):

        import operator

        for cur_trait, cur_dict_score in self.trait_accuracy_dict.iteritems():
            logging.info('')
            logging.info('Best accuracy for personality trait: ' + str(cur_trait))
            cur_trait_tuple = sorted(cur_dict_score.items(),
                                      key=operator.itemgetter(1))  # sort aspect by their common
            cur_trait_tuple.reverse()
            logging.info('Best configuration: ' + str(cur_trait_tuple[0][0]))
            logging.info('Best Accuracy: ' + str(round(cur_trait_tuple[0][1], 3)))

            logging.info('All: ' + str(cur_trait_tuple))

        for cur_trait, cur_dict_score in self.trait_auc_dict.iteritems():
            logging.info('')
            logging.info('Best AUC for personality trait: ' + str(cur_trait))
            cur_trait_tuple = sorted(cur_dict_score.items(),
                                      key=operator.itemgetter(1))  # sort aspect by their common
            cur_trait_tuple.reverse()
            logging.info('Best configuration: ' + str(cur_trait_tuple[0][0]))
            logging.info('Best AUC: ' + str(round(cur_trait_tuple[0][1], 3)))

            logging.info('All: ' + str(cur_trait_tuple))

        logging.info('Num models investigate: ' + str(self.num_models))


        return


def main(file_directory, log_dir, eli_5_dir, plot_dir, load_already_split_df_bool, model_type):

    wrapper_obj = WrapperModel(file_directory, log_dir, eli_5_dir, plot_dir, load_already_split_df_bool, model_type)  # create object and variables
    wrapper_obj.init_debug_log()                            # init log file
    wrapper_obj.run_experiments()


if __name__ == '__main__':

    file_directory = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/predict_personality_from_descriptions/dataset/2018-02-26 11:02:35'
    load_already_split_df_bool = False

    file_directory = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/predict_personality_from_descriptions/dataset/2018-03-13 08:44:49_ratio_0.2_0.8'
    load_already_split_df_bool = True
    model_type = 'n-gram-rr'     # 'n-gram-rr'        # 'lstm'

    log_dir = 'log/'
    eli_5_dir = 'eli5_explanation/'
    plot_dir = 'plot/'

    main(file_directory, log_dir, eli_5_dir, plot_dir, load_already_split_df_bool, model_type)
