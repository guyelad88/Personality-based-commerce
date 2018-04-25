from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from eli5_methods import StaticClass
import sklearn
import scipy


class PredictDescriptionModel:

    def __init__(self,
                 file_directory,
                 log_dir,
                 eli_5_dir,
                 plot_dir,
                 load_already_split_df_bool,
                 model_type,                    # 'keras' / 'n-gram-rr'
                 cur_estimator,
                 cur_model,
                 cur_ngram,
                 cur_norm,
                 use_idf,
                 smooth_idf,
                 sublinear_tf,
                 min_df,
                 max_df,
                 cur_key,
                 cur_time,
                 lstm_parameters=dict,
                 equal_labels=True):

        # file arguments
        self.file_directory = file_directory    # directory contain all data for all traits
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.load_already_split_df_bool = load_already_split_df_bool    # load split df or merge (and split inside)
        self.equal_labels = equal_labels        # number of items in each group will be equal - min(i1,i2)
        self.eli_5_dir = eli_5_dir              # directory to contain eli 5 html files
        self.cur_key = cur_key                  # model sum parameters
        self.verbose_flag = True
        self.model_type = model_type            # lstm / n-gram-rr
        self.cur_estimator = cur_estimator
        self.cur_model = cur_model
        self.cur_ngram = cur_ngram
        self.cur_norm = cur_norm

        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self.min_df = min_df        # int - minimum number of doc, float - min prop of docs
        self.max_df = max_df        # float - max prop of doc he appears in

        self.cur_time = cur_time
        # from time import gmtime, strftime
        # self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.create_data_set_bool = False

        self.lstm_parameters = lstm_parameters
        '''{
            'max_features': 200000,
            'maxlen': 200,  # cut texts after this number of words (among top max_features most common words)
            'batch_size': 32,
            'embedding_size': 32,
            'num_epoch': 2,
            'dropout': 0.2,  # 0.2
            'recurrent_dropout': 0.2,  # 0.2
            'tensor_board_bool': False
        }'''

        self.predict_personality_accuracy = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

        self.predict_personality_AUC = {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }

        self.traits_list = [
            'agreeableness',
            'extraversion',
            'openness',
            'conscientiousness',
            'neuroticism'
        ]

        self.df = list()            # contain
        self.train_df = list()
        self.test_df = list()
        self.count_vec = None       # skleran text obj

        self.text_list_list_p = list()
        self.text_list_list_q = list()
        self.len_p = np.float
        self.len_q = np.float

        self.X_train = list()
        self.X_test = list()
        self.y_train = list()
        self.y_test = list()

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

    # load csv into df
    def run_experiment(self):

        import os

        # run over input directory contain 5 sub-folder, folder to each trait
        for root, dirnames, filenames in os.walk(self.file_directory):
            for cur_trait in dirnames:

                if cur_trait not in self.traits_list:
                    continue

                logging.info('')
                logging.info('Build classifier for trait: ' + str(cur_trait))

                self.run_experiment_trait(cur_trait)

        self.sum_write_results()
        return

    def run_experiment_trait(self, cur_trait):

        # load csv already split to train/test or build them
        if self.load_already_split_df_bool:
            logging.info('load train-test from cache')
            self.load_train_test(cur_trait)
        else:
            logging.info('create test-train')
            self.load_data_from_csv(cur_trait)
            self.create_data_frame()
            self.split_train_test()

        if self.model_type == 'lstm':
            self.run_model_lstm_keras(cur_trait)

        elif self.model_type == 'n-gram-rr':
            self.create_count_vec_obj()
            self.create_x_y_train_test_data(cur_trait)
            regr, test_score, auc_score, fpr, tpr, thresholds = self.run_model(cur_trait)
            self.create_roc_plot(auc_score, test_score, fpr, tpr, cur_trait)
            self.run_eli_5_analysis(regr, test_score, auc_score, cur_trait)

        return

    # load data from two files (H/L)
    def load_data_from_csv(self, cur_trait):

        import pickle
        import os
        # for each trait
        for root_in, dirnames_in, filenames_in in os.walk(self.file_directory + '/' + str(cur_trait)):
            for idx, file_name_suffix in enumerate(filenames_in):

                cur_file = self.file_directory + '/' + str(cur_trait) + '/' + str(file_name_suffix)

                if 'high' in file_name_suffix:
                    with open(cur_file, 'rb') as fp:
                        text_list_list_p = pickle.load(fp)
                        self.text_list_list_p = text_list_list_p  # all items in p
                        self.len_p = np.float(len(text_list_list_p))
                        logging.info('P #of items descriptions: ' + str(self.len_p))

                elif 'low' in file_name_suffix:
                    with open(cur_file, 'rb') as fp:
                        text_list_list_q = pickle.load(fp)
                        self.text_list_list_q = text_list_list_q  # all items in q
                        self.len_q = np.float(len(text_list_list_q))
                        logging.info('Q #of items descriptions: ' + str(self.len_q))
                else:
                    raise ('file name unknown: ' + str(file_name_suffix))

        return

    # create data set contain descriptions and corresponds labels
    def create_data_frame(self):

        if self.equal_labels:
            equal_num_desc = min(len(self.text_list_list_p), len(self.text_list_list_q))
            logging.info('create two equal groups: ' + str(equal_num_desc))

        # labels: 1 - high, 0 - low
        self.df = pd.DataFrame(columns=["item_desc", "label"])
        for idx, cur_desc in enumerate(self.text_list_list_p):
            if self.equal_labels and idx >= equal_num_desc:
                continue
            self.df = self.df.append({
                "item_desc": cur_desc,
                "label": 1
            }, ignore_index=True)

        for idx, cur_desc in enumerate(self.text_list_list_q):
            if self.equal_labels and idx >= equal_num_desc:
                continue
            self.df = self.df.append({
                "item_desc": cur_desc,
                "label": 0
            }, ignore_index=True)
        return

    # create count_vec object (currently n-gram/tf-idf)
    def create_count_vec_obj(self):

        if self.cur_model == 'n-gram':
            self.count_vec = CountVectorizer(
                ngram_range=self.cur_ngram,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english',
                lowercase=True
            )
        elif self.cur_model == 'tf-idf':
            self.count_vec = TfidfVectorizer(
                ngram_range=self.cur_ngram,
                min_df=self.min_df,
                max_df=self.max_df,
                norm=self.cur_norm,
                use_idf=self.use_idf,
                smooth_idf=self.smooth_idf,
                sublinear_tf=self.sublinear_tf,
                stop_words='english',
                lowercase=True)
        else:
            logging.info('unknown model type: ' + str(self.cur_model))
            raise()

        return

    # split to train/test df
    def split_train_test(self):

        from sklearn.model_selection import train_test_split
        test_size = 0.2

        self.train_df, self.test_df = train_test_split(
            self.df,
            stratify=self.df['label'],
            test_size=test_size
        )

        if self.create_data_set_bool:
            self.create_data_set(test_size)

        return

    # load train/test df
    def load_train_test(self, cur_trait):

        cur_folder = self.file_directory + '/' + str(cur_trait) + '/'

        self.train_df = pd.read_csv(cur_folder + 'train.csv')
        self.test_df = pd.read_csv(cur_folder + 'test.csv')

        logging.info('Load data set from cache')
        logging.info('train size: ' + str(self.train_df.shape[0]) + ', ratio: ' + str(self.train_df['label'].mean()))
        logging.info('test size: ' + str(self.test_df.shape[0]) + ', ratio: ' + str(self.test_df['label'].mean()))

        return

    # split data set into different objects - x,y,train,test
    def create_x_y_train_test_data(self, cur_trait):

        import numpy
        x_val = self.train_df['item_desc'].values
        self.X_train = self.count_vec.fit_transform(x_val)
        self.X_test = self.count_vec.transform(self.test_df['item_desc'].values)        # transfrom using train data

        self.y_train = self.train_df['label'].tolist()
        self.y_test = self.test_df['label'].tolist()

        high_train = max(float(sum(self.y_train)) / float(len(self.y_train)), 1 - float(sum(self.y_train)) / float(len(self.y_train)))
        high_test = max(float(sum(self.y_test)) / float(len(self.y_test)), 1 - float(sum(self.y_test)) / float(len(self.y_test)))

        # logging.info('1 label amount: ' + str(self.df['label'].value_counts()[1]))
        # logging.info('0 label amount: ' + str(self.df['label'].value_counts()[0]))

        logging.info('train - high proportion: ' + str(high_train))
        logging.info('test - high proportion: ' + str(high_test))

        assert (self.X_train.shape[1] == self.X_test.shape[1])
        return

    # create and run model
    def run_model(self, cur_trait):

        from sklearn import linear_model
        from sklearn import svm
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import NuSVC
        import numpy as np
        from sklearn import metrics

        if self.cur_estimator == 'logistic_cv':
            regr = linear_model.LogisticRegressionCV()
        elif self.cur_estimator == 'logistic':
            regr = linear_model.LogisticRegression()
        elif self.cur_estimator == 'AdaBoost':
            regr = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                      algorithm="SAMME",
                                      n_estimators=200)
            # regr = linear_model.LogisticRegression()
        elif self.cur_estimator == 'NuSVC':
            regr = NuSVC(probability=True)
        else:
            raise('Unknown estimator type: ' + str(self.cur_estimator))

        # regr = svm.SVC(kernel='linear', probability=True)

        regr.fit(self.X_train, self.y_train)

        train_score = regr.score(self.X_train, self.y_train)
        test_score = regr.score(self.X_test, self.y_test)
        y_pred = regr.predict(self.X_test)
        y_pred_prob = regr.predict_proba(self.X_test)

        logging.info('Number of parameters: ' + str(self.X_train.shape[1]))
        logging.info('train samples: ' + str(self.X_train.shape[0]))
        logging.info('test samples: ' + str(self.X_test.shape[0]))
        logging.info('train_score: ' + str(round(train_score, 3)))
        logging.info('test_score: ' + str(round(test_score, 3)))
        logging.info('majority_score: ' + str(round(max(
            float(sum(self.y_test)) / float(len(self.y_test)),
            1 - (float(sum(self.y_test)) / float(len(self.y_test)))), 3)))
        logging.info('model regr: ' + str(regr))

        prob_list = list()
        for (x, y), value in np.ndenumerate(y_pred_prob):
            if y == 1:
                prob_list.append(value)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, prob_list)  # , pos_label=2)
        auc_score = round(metrics.auc(fpr, tpr), 3)
        logging.info('AUC: ' + str(auc_score))

        self.predict_personality_accuracy[cur_trait] = test_score
        self.predict_personality_AUC[cur_trait] = auc_score
        return regr, test_score, auc_score, fpr, tpr, thresholds

    # run lstm model with embedding using Keras platform
    def run_model_lstm_keras(self, cur_trait):

        from classifier_lstm import PredictDescriptionModelLSTM
        logging.info('')
        logging.info('Run LSTM on Keras')

        lstm_obj = PredictDescriptionModelLSTM(
            self.file_directory,
            logging,
            self.cur_time,
            self.train_df['item_desc'],
            self.train_df['label'],
            # self.y_train,
            self.test_df['item_desc'],
            self.test_df['label'],
            # self.y_test,
            self.lstm_parameters
        )
        test_score, test_accuracy = lstm_obj.run_LSTM_model()
        self.predict_personality_accuracy[cur_trait] = test_accuracy
        logging.info('finish LSTM model')
        logging.info('trait: ' + str(cur_trait) + ', accuracy: ' + str(test_accuracy))
        import time
        time.sleep(5)
        return

    # create roc plots
    def create_roc_plot(self, auc_score, test_score, fpr, tpr, cur_trait):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(str(cur_trait) + ': test amount ' + str(self.X_test.shape[0]) + ', test prop ' + str(
            round(sum(self.y_test) / len(self.y_test), 2)))
        plt.legend(loc="lower right")

        plot_name = str(round(auc_score, 2)) + '_AUC_accuracy_' + str(round(test_score, 2)) + '_trait_' + \
                    str(cur_trait) + '_' + str(self.cur_key) + '.png'

        import os
        if not os.path.exists(self.plot_dir + '/' + str(self.cur_time) + '/'):
            os.makedirs(self.plot_dir + '/' + str(self.cur_time) + '/')

        plot_path = self.plot_dir + '/' + str(self.cur_time) + '/' + plot_name
        plt.savefig(plot_path, bbox_inches='tight')

        plt.close()
        return

    # run eli 5 library analysis and save results
    def run_eli_5_analysis(self, regr, test_score, auc_score, cur_trait):
        # use eli5 library to explain weight and predictions
        try:
            StaticClass._logit_model_eli5_explain_weights('logistic', self.cur_key, cur_trait, test_score, regr,
                                                          self.count_vec, self.eli_5_dir, logging, self.cur_time)

            StaticClass._logit_model_eli5_explain_prediction('logistic', self.cur_key, cur_trait, test_score, auc_score,
                                                             regr, self.count_vec, self.test_df['item_desc'].values,
                                                             self.X_test,
                                                             self.y_test, self.eli_5_dir, logging, self.cur_time, k=10)
        except:
            logging.info('eli 5 implementation failed')
            pass
        return

    # write to log file best results
    def sum_write_results(self):
        d_view = [(v, k) for k, v in self.predict_personality_accuracy.iteritems()]
        d_view.sort(reverse=True)  # natively sort tuples by first element
        logging.info('')
        for v, k in d_view:
            logging.info("%s: %f" % (k, v))
        return

    # save csv test train (use for compare between models)
    def create_data_set(self, test_size):
        import os
        output_path = 'dataset' + '/' + str(self.cur_time) + '_ratio_' + str(test_size) + '_' + str(1-test_size) + '/' + str(cur_trait)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.train_df.to_csv(output_path + '/' + 'train' + '.csv')
        self.test_df.to_csv(output_path + '/' + 'test' + '.csv')

        logging.info('save df to path: ' + str(output_path) + '/' + 'train.csv')
        logging.info('save df to path: ' + str(output_path) + '/' + 'test.csv')
        return


def main(file_directory, log_dir):

    raise('cuurenlty only run from wrapper classifer')

    pred_desc_obj = PredictDescriptionModel(file_directory, log_dir)    # create object and variables

    pred_desc_obj.init_debug_log()                          # init log file
    pred_desc_obj.run_experiment()                           # load data set
    # logistic_obj.build_data_set()                      # build data set - merge data


if __name__ == '__main__':

    file_directory = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/predict_personality_from_descriptions/dataset/2018-02-26 11:02:35'
    log_dir = 'log/'
    main(file_directory, log_dir)
