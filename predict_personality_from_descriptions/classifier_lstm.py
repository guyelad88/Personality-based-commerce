from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from eli5_methods import StaticClass
import sklearn
import scipy


class PredictDescriptionModelLSTM:
    def __init__(self,
                 file_directory,
                 logging,
                 cur_time,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 network_dict
                 ):

        # file arguments
        self.file_directory = file_directory  # directory contain all data for all traits
        self.logging = logging
        self.cur_time = cur_time

        self.x_train = x_train      # list of string - string per item description (before insert into count vec)
        self.x_test = x_test        # list of string - string per item description (before insert into count vec)
        self.y_train = y_train
        self.y_test = y_test

        self.x_train_sequence = None
        self.x_test_sequence = None

        self.max_features = network_dict['max_features']               # 20000
        self.maxlen = network_dict['maxlen']                           # 200 (item description maximum length)
        self.batch_size = network_dict['batch_size']                   # 16
        self.embedding_size = network_dict['embedding_size']           # 16
        self.num_epoch = network_dict['num_epoch']                     # 20
        self.dropout = network_dict['dropout']                         # 0.2
        self.recurrent_dropout = network_dict['recurrent_dropout']     # 0.2
        self.tensor_board_bool = network_dict['tensor_board_bool']
        self.max_num_words = network_dict['max_num_words']

        self.logging.info('')
        self.logging.info('Network parameters: ')
        for param, value in network_dict.iteritems():
            self.logging.info('Parameter: ' + str(param) + ', Val: ' + str(value))

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

    def run_LSTM_model(self):

        '''import keras
        from keras.preprocessing import text
        from keras.preprocessing import sequence
        from keras.models import Sequential
        from keras.layers import Dense, Embedding
        from keras.layers import LSTM
        from keras.datasets import imdb'''

        self.prepare_data()         # tokenizer, fit
        test_score, test_accuracy = self.run_LSTM()

        return test_score, test_accuracy
        '''

        
        from keras.preprocessing.text import one_hot
        from keras.preprocessing.text import text_to_word_sequence

        text = ['the dog is amazing, like a big cat or even a rat', 'i love hagar, she amazing']

        ker_obj = keras.preprocessing.text.Tokenizer(num_words=None,
                                                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                     lower=True,
                                                     split=" ",
                                                     char_level=False,
                                                     oov_token=None)
        docs = ['Well done!',
                'Good work',
                'Great effort',
                'nice work',
                'Excellent!']
        # create the tokenizer
        t = keras.preprocessing.text.Tokenizer()
        # fit the tokenizer on the documents
        t.fit_on_texts(self.train_df)
        a = t.texts_to_sequences(self.train_df)
        print(a)
        # output = ker_obj.texts_to_sequences(text)


        # define the document
        text_list = ['The quick brown fox jumped over the lazy dog.', 'I love and cat']
        # estimate the size of the vocabulary
        res_list = list()
        for idx_sen, text in enumerate(text_list):
            words = set(text_to_word_sequence(text))
            vocab_size = 10                             # len(words)
            print(vocab_size)
            # integer encode the document
            result = one_hot(text, round(vocab_size * 1.3))
            print(result)
            res_list.append(result)

        a = 5


        from keras.preprocessing.text import Tokenizer
        # define 5 documents
        docs = ['Well done!',
                'Good work',
                'Great effort',
                'nice work',
                'Excellent!']
        # create the tokenizer
        t = Tokenizer()
        # fit the tokenizer on the documents
        t.fit_on_texts(docs)



        from keras.preprocessing.text import one_hot
        from keras.preprocessing.text import text_to_word_sequence
        # define the document
        text = ['The quick brown fox jumped over the lazy dog.', 'i love deep learning and fox']
        # estimate the size of the vocabulary
        words = set(text_to_word_sequence(text))
        vocab_size = len(words)
        print(vocab_size)
        # integer encode the document
        result = one_hot(text, round(vocab_size * 1.3))
        print(result)


        keras.preprocessing.text.text_to_word_sequence(text,
                                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                       lower=True,
                                                       split=" ")
        check_one_hot = text.one_hot(text,
                                 n=max_features,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True,
                                 split=" ")
        print('Loading data...')
        
        return'''

    # fit to
    def prepare_data(self):

        from keras.preprocessing import text
        # from keras.preprocessing.text import one_hot
        t = text.Tokenizer(num_words=self.max_num_words,    # None/10000,  # None, from 17798
                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                           lower=True,
                           split=" ",
                           char_level=False,
                           oov_token=None)

        # fit the tokenizer on the documents
        t.fit_on_texts(self.x_train)

        # self.logging.info('# docs: ' + str(t.document_count))
        # self.logging.info('t word index: ' + str(t.word_index))
        # self.logging.info('t word counts: ' + str(t.word_counts))
        # self.logging.info('t word docs: ' + str(t.word_docs))

        self.x_train = t.texts_to_sequences(self.x_train)
        self.x_test = t.texts_to_sequences(self.x_test)

        return

    def run_LSTM(self):

        from keras.preprocessing import text
        from keras.preprocessing import sequence
        from keras.models import Sequential
        from keras.layers import Dense, Embedding
        from keras.layers import LSTM
        from keras.callbacks import TensorBoard

        '''
        max_features = 20000
        maxlen = 200  # cut texts after this number of words (among top max_features most common words)
        batch_size = 16
        embedding_size = 16
        num_epoch = 20
        dropout = 0.2               # o.2
        recurrent_dropout = 0.2     # 0.2
        '''

        self.logging.info(str(len(self.x_train)) + ' train sequences')
        self.logging.info(str(len(self.x_test)) + ' test sequences')

        self.logging.info('Pad sequences (samples x time)')

        self.x_train = sequence.pad_sequences(
            self.x_train,
            maxlen=self.maxlen
        )

        self.x_test = sequence.pad_sequences(
            self.x_test,
            maxlen=self.maxlen
        )

        self.logging.info('x_train shape: ' + str(self.x_train.shape))
        self.logging.info('x_test shape: ' + str(self.x_test.shape))
        self.logging.info('y_train shape: ' + str(self.y_train.shape))
        self.logging.info('y_test shape: ' + str(self.y_test.shape))

        self.logging.info('Build model...')

        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_size))
        model.add(LSTM(self.embedding_size, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.logging.info(model.summary())

        tensor_board_dir = './Graph/' + str(self.cur_time)
        import os
        if not os.path.exists(tensor_board_dir):
            os.makedirs(tensor_board_dir)

        self.logging.info('Train...')

        if self.tensor_board_bool:
            tensor_board = TensorBoard(
                log_dir=tensor_board_dir,    # './Graph',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )

            model.fit(self.x_train,
                      self.y_train,
                      batch_size=self.batch_size,
                      epochs=self.num_epoch,
                      validation_data=(self.x_test, self.y_test),
                      shuffle=True,
                      callbacks=[tensor_board]
                      )
        else:
            model.fit(self.x_train,
                      self.y_train,
                      batch_size=self.batch_size,
                      epochs=self.num_epoch,
                      validation_data=(self.x_test, self.y_test),
                      shuffle=True
                      )

        test_score, test_accuracy = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        self.logging.info('Test score: ' + str(test_score))
        self.logging.info('Test accuracy: ' + str(test_accuracy))
        return test_score, test_accuracy


def main(file_directory, log_dir):

    raise ('cuurenlty only run from wrapper classifer')

    pred_desc_obj = PredictDescriptionModelLSTM(file_directory, log_dir)  # create object and variables

    pred_desc_obj.init_debug_log()  # init log file
    pred_desc_obj.run_experiment()  # load data set
    # logistic_obj.build_data_set()                      # build data set - merge data


if __name__ == '__main__':
    file_directory = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/predict_personality_from_descriptions/dataset/2018-02-26 11:02:35'
    log_dir = 'log/'
    main(file_directory, log_dir)
