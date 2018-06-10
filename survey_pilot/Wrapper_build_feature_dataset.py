from __future__ import print_function
import logging
from build_feature_dataset import CalculateScore        # class which extract data set from input files

class Wrapper:

    def __init__(self, participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
                 dir_logistic_results, dict_feature_flag, model_method, user_type, l_limit, h_limit, k_best_list, threshold_list, penalty, bool_slice_gap_percentile):

        # file arguments
        self.participant_file = participant_file
        self.item_aspects_file = item_aspects_file
        self.purchase_history_file = purchase_history_file
        self.valid_users_file = valid_users_file
        self.dir_analyze_name = dir_analyze_name
        self.dir_logistic_results = dir_logistic_results

        self.user_type = user_type                  # which user to keep in model 'all'/'cf'/'ebay-tech'
        self.model_method = model_method            # 'linear'/'logistic'
        self.k_best_list = k_best_list
        self.threshold_list = threshold_list
        self.penalty = penalty
        self.bool_slice_gap_percentile = bool_slice_gap_percentile
        self.h_limit = h_limit
        self.l_limit = l_limit

        self.threshold_purchase = 1
        self.bool_normalize_features = True
        self.cur_C = 10
        self.cur_penalty = 'l2'

        self.split_test = False
        self.normalize_traits = True    # normalize traits to 0-1 (divided by 5)

        # bool values for which feature will be in the model (before sliced by in the selectKbest)
        self.time_purchase_ratio_feature_flag = dict_feature_flag['time_purchase_ratio_feature_flag']
        self.time_purchase_meta_feature_flag = dict_feature_flag['time_purchase_meta_feature_flag']
        self.vertical_ratio_feature_flag = dict_feature_flag['vertical_ratio_feature_flag']
        self.purchase_percentile_feature_flag = dict_feature_flag['purchase_percentile_feature_flag']
        self.user_meta_feature_flag = dict_feature_flag['user_meta_feature_flag']
        self.aspect_feature_flag = dict_feature_flag['aspect_feature_flag']

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.verbose_flag = True

        self.plot_directory = self.dir_logistic_results + str(self.model_method) + '_results/'
        if self.model_method == 'logistic':
            self.plot_directory += 'gap_' + str(self.h_limit) + '_' + str(self.l_limit) + '_time_' + str(self.cur_time) + '/'
        elif self.model_method == 'linear':
            self.plot_directory += 'time_' + str(self.cur_time) + '/'
        else:
            raise('unknown method type')

        an = self
        self.input_attr_dict = vars(an)  # save attribute to show them later

        import os
        if not os.path.exists(self.plot_directory):
            os.makedirs(self.plot_directory)

        # "regular" accuracy results
        self.openness_score_list = list()
        self.conscientiousness_score_list = list()
        self.extraversion_score_list = list()
        self.agreeableness_score_list = list()
        self.neuroticism_score_list = list()

        # CV results
        self.openness_cv_score_list = list()
        self.conscientiousness_cv_score_list = list()
        self.extraversion_cv_score_list = list()
        self.agreeableness_cv_score_list = list()
        self.neuroticism_cv_score_list = list()

        # ROC results
        self.openness_score_roc_list = list()
        self.conscientiousness_score_roc_list = list()
        self.extraversion_score_roc_list = list()
        self.agreeableness_score_roc_list = list()
        self.neuroticism_score_roc_list = list()

        self.max_score = {
            "openness": {'i': -1, 'score': 0.0},
            "agreeableness": {'i': -1, 'score': 0.0},
            "conscientiousness": {'i': -1, 'score': 0.0},
            "extraversion": {'i': -1, 'score': 0.0},
            "neuroticism": {'i': -1, 'score': 0.0},
        }

        self.i = int
        self.k_best = int

    ############################# validation functions #############################

    def init_debug_log(self):
        import logging

        lod_file_name = '/Users/gelad/Personality-based-commerce/BFI_results/log/' + 'wrapper_logistic_regression' + str(self.cur_time) + '.log'

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

    # check if input data is valid
    def check_input(self):

        if self.user_type not in ['all', 'cf', 'ebay-tech']:
            raise ('unknown user_type')

        if self.model_method not in ['linear', 'logistic']:
            raise ('unknown model_method')

        if len(self.k_best_list)<1:
            raise ('empty k_best_list')

        if len(self.threshold_list)<1:
            raise ('empty threshold_list')

        if len(self.penalty)<1:
            raise ('empty penalty')

        if self.h_limit < 0 or self.h_limit > 1:
            raise('h_limit must be a float between 0 to 1')

        if self.l_limit < 0 or self.l_limit > 1:
            raise('l_limit must be a float between 0 to 1')

        if self.l_limit >= self.h_limit:
            raise ('l_limit must be smaller than h_limit')

        return

    # log class arguments
    def log_attribute_input(self):

        logging.info('')
        logging.info('Class arguments')
        import collections
        od = collections.OrderedDict(sorted(self.input_attr_dict.items()))
        for attr, attr_value in od.iteritems():
            if type(attr_value) is list and len(attr_value)==0:
                continue
            logging.info('Attribute: ' + str(attr) + ', Value: ' + str(attr_value))
        logging.info('')

        return

    ############################# main functions #############################
    '''
    table of contents (chronological order)
    1. run models - router correspond to model received 
    2. wrapper_experiments_logistic/linear - run configurations 
    3. run_experiments - for a specific configuration call inner class
        3.1 build data set (mainly using threshold + features defined)
        3.2 run model and store results
    '''
    # which model ('linear'/'logistic')
    def run_models(self):

        self.check_input()          # check if input argument are valid
        self.log_attribute_input()  # log arguments for all model

        if self.model_method == 'linear':
            self.wrapper_experiments_linear()
        elif self.model_method == 'logistic':
            self.wrapper_experiments_logistic()
        return

    # run if we choose logistic method
    # run all possible configurations given inputs
    def wrapper_experiments_logistic(self):

        for cur_penalty in self.penalty:
            for k_best in self.k_best_list:

                # reset evaluation list. list target is to compare between configurations with same K_best and penalty
                self.openness_score_list = list()
                self.conscientiousness_score_list = list()
                self.extraversion_score_list = list()
                self.agreeableness_score_list = list()
                self.neuroticism_score_list = list()

                self.openness_score_roc_list = list()
                self.conscientiousness_score_roc_list = list()
                self.extraversion_score_roc_list = list()
                self.agreeableness_score_roc_list = list()
                self.neuroticism_score_roc_list = list()

                self.openness_cv_score_list = list()
                self.conscientiousness_cv_score_list = list()
                self.extraversion_cv_score_list = list()
                self.agreeableness_cv_score_list = list()
                self.neuroticism_cv_score_list = list()

                for threshold_purchase in self.threshold_list:

                    self.threshold_purchase = threshold_purchase
                    self.k_best = k_best
                    self.cur_penalty = cur_penalty      # TODO change

                    logging.info('')
                    logging.info('')
                    logging.info('############################# run new configuration #############################')
                    logging.info('Current configuration: Penalty: ' + str(cur_penalty) + ', Threshold: ' + str(threshold_purchase) + ', k_best: ' + str(k_best) )

                    calculate_obj = self.run_experiments()     # run configuration

                    # store result correspond to whether we split data or not
                    if self.split_test:
                        # test score
                        self.openness_score_list.append(calculate_obj.logistic_regression_accuracy['openness'])
                        self.conscientiousness_score_list.append(calculate_obj.logistic_regression_accuracy['conscientiousness'])
                        self.extraversion_score_list.append(calculate_obj.logistic_regression_accuracy['extraversion'])
                        self.agreeableness_score_list.append(calculate_obj.logistic_regression_accuracy['agreeableness'])
                        self.neuroticism_score_list.append(calculate_obj.logistic_regression_accuracy['neuroticism'])

                        # roc score
                        self.openness_score_roc_list.append(calculate_obj.logistic_regression_roc['openness'])
                        self.conscientiousness_score_roc_list.append(calculate_obj.logistic_regression_roc['conscientiousness'])
                        self.extraversion_score_roc_list.append(calculate_obj.logistic_regression_roc['extraversion'])
                        self.agreeableness_score_roc_list.append(calculate_obj.logistic_regression_roc['agreeableness'])
                        self.neuroticism_score_roc_list.append(calculate_obj.logistic_regression_roc['neuroticism'])

                    # CV score
                    self.openness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['openness'])
                    self.conscientiousness_cv_score_list.append(
                        calculate_obj.logistic_regression_accuracy_cv['conscientiousness'])
                    self.extraversion_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['extraversion'])
                    self.agreeableness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['agreeableness'])
                    self.neuroticism_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['neuroticism'])

                # plot results
                # if we split test+train we present test score + ROC curve
                # else we only present CV score without TODO ROC curve
                if self.split_test:
                    self.plot_traits_accuracy_versus_threshold(cur_penalty, k_best)
                    self.plot_traits_roc_versus_threshold(cur_penalty, k_best)
                else:
                    self.plot_traits_accuracy_versus_threshold_CV(cur_penalty, k_best)

    def wrapper_experiments_linear(self):

        self.k_best_list = [15, 12, 8, 5]
        self.threshold_list = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        penalty = ['l1', 'l2']

        # for cur_C in C:
        for cur_penalty in penalty:

            logging.info('Penalty: ' + str(cur_penalty))
            for k_best in self.k_best_list:

                self.openness_score_mae_list = list()
                self.conscientiousness_score_mae_list = list()
                self.extraversion_score_mae_list = list()
                self.agreeableness_score_mae_list = list()
                self.neuroticism_score_mae_list = list()

                self.openness_score_pearson_list = list()
                self.conscientiousness_score_pearson_list = list()
                self.extraversion_score_pearson_list = list()
                self.agreeableness_score_pearson_list = list()
                self.neuroticism_score_pearson_list = list()

                '''self.openness_cv_score_list = list()
                self.conscientiousness_cv_score_list = list()
                self.extraversion_cv_score_list = list()
                self.agreeableness_cv_score_list = list()
                self.neuroticism_cv_score_list = list()'''

                for threshold_purchase in self.threshold_list:
                    self.threshold_purchase = threshold_purchase
                    self.k_best = k_best
                    self.cur_penalty = cur_penalty
                    logging.info('Penalty: ' + str(cur_penalty) + ', Threshold: ' + str(threshold_purchase))

                    calculate_obj = self.run_experiments()

                    # cur_key = 'C_' + str(cur_C) + '_Penalty_' + str(cur_penalty) + '_Threshold_' + str(threshold_purchase)
                    cur_key = '_Penalty_' + str(cur_penalty) + '_Threshold_' + str(threshold_purchase)


                    # mae score
                    self.openness_score_mae_list.append(calculate_obj.linear_regression_mae['openness'])
                    self.conscientiousness_score_mae_list.append(calculate_obj.linear_regression_mae['conscientiousness'])
                    self.extraversion_score_mae_list.append(calculate_obj.linear_regression_mae['extraversion'])
                    self.agreeableness_score_mae_list.append(calculate_obj.linear_regression_mae['agreeableness'])
                    self.neuroticism_score_mae_list.append(calculate_obj.linear_regression_mae['neuroticism'])

                    # pearson score
                    self.openness_score_pearson_list.append(calculate_obj.linear_regression_pearson['openness'])
                    self.conscientiousness_score_pearson_list.append(calculate_obj.linear_regression_pearson['conscientiousness'])
                    self.extraversion_score_pearson_list.append(calculate_obj.linear_regression_pearson['extraversion'])
                    self.agreeableness_score_pearson_list.append(calculate_obj.linear_regression_pearson['agreeableness'])
                    self.neuroticism_score_pearson_list.append(calculate_obj.linear_regression_pearson['neuroticism'])

                    '''# CV score
                    self.openness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['openness'])
                    self.conscientiousness_cv_score_list.append(
                        calculate_obj.logistic_regression_accuracy_cv['conscientiousness'])
                    self.extraversion_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['extraversion'])
                    self.agreeableness_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['agreeableness'])
                    self.neuroticism_cv_score_list.append(calculate_obj.logistic_regression_accuracy_cv['neuroticism'])'''


                self.plot_traits_mae_versus_threshold_linear(cur_penalty, k_best)
                self.plot_traits_pearson_versus_threshold_linear(cur_penalty, k_best)

    # run experiments for a giving arguments
    def run_experiments(self):
        calculate_obj = CalculateScore(participant_file, item_aspects_file, purchase_history_file, valid_users_file,
                                       dir_analyze_name, self.threshold_purchase, self.bool_slice_gap_percentile,
                                       self.bool_normalize_features, self.cur_C, self.cur_penalty,
                                       self.time_purchase_ratio_feature_flag, self.time_purchase_meta_feature_flag,
                                       self.vertical_ratio_feature_flag, self.purchase_percentile_feature_flag,
                                       self.user_meta_feature_flag, self.aspect_feature_flag ,self.h_limit, self.l_limit, self.k_best,
                                       self.plot_directory, self.user_type, self.normalize_traits)

        # calculate_obj.init_debug_log()  # init log file
        calculate_obj.load_clean_csv_results()                  # load data set
        calculate_obj.clean_df()                                # clean df - e.g. remain valid users only
        calculate_obj.create_feature_list()                     # create x_feature

        calculate_obj.insert_gender_feature()                   # add gender feature
        calculate_obj.remove_except_cf()                        # remove not CF participants
        calculate_obj.extract_user_purchase_connection()        # insert purchase and vertical type to model
        calculate_obj.extract_item_aspect()                     # add features of dominant item aspect

        calculate_obj.normalize_personality_trait()             # normalize trait to 0-1 scale (div by 5)

        # important!! after cut users by threshold
        calculate_obj.cal_participant_percentile_traits_values()  # calculate average traits and percentile value

        calculate_obj.insert_money_feature()                    # add feature contain money issue
        calculate_obj.insert_time_feature()                     # add time purchase feature

        if self.model_method == 'linear':
            calculate_obj.calculate_linear_regression()         # predict values 0-1, MAE and Pearson
        elif self.model_method == 'logistic':
            calculate_obj.calculate_logistic_regression()       # predict traits H or L

        return calculate_obj

    ############################# visualization functions #############################

    # plot traits accuracy versus - logistic
    def plot_traits_accuracy_versus_threshold(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt

        try:
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 2, 1)

            plt.figure(figsize=(10, 6))
            plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_cv_score_list, '.r:', label='openness CV')

            plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_cv_score_list, '.b:', label='conscientiousness CV')

            plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_cv_score_list, '.g:', label='extraversion CV')

            plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_cv_score_list, '.m:', label='agreeableness CV')

            plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_cv_score_list, '.c:', label='neuroticism CV')

            max_open = max(max(self.openness_score_list),
                           max(self.conscientiousness_score_list),
                           max(self.extraversion_score_list),
                           max(self.agreeableness_score_list),
                           max(self.neuroticism_score_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'traits test accuracy vs. amount purchase threshold \n'
            title += ' Max: ' + str(round(max_open, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(cur_penalty)\
                     + ' Gap:' + str(self.h_limit) + '-' + str(self.l_limit)

            plt.title(title)
            plt.ylabel('Test accuracy')
            plt.xlabel('Threshold purchase amount')
            # plt.ylim(0.4, 1)
            # plot_name = cur_directory + 'logistic_C=' + str(cur_C) + '_penalty=' \
            #             + str(cur_penalty) + '_max=' + str(round(max_open, 2)) + '_gap=' + str(
            #     bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '.png'
            plot_name = str(round(max_open, 2)) + '_Accuracy_k=' + str(k_best) + '_penalty=' + str(cur_penalty) + '_gap=' + str(
                self.h_limit) + '_' + str(self.l_limit) + '_max=' + str(round(max_open, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            logging.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass

        return

    # plot traits accuracy versus - logistic
    def plot_traits_accuracy_versus_threshold_CV(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt

        try:
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 2, 1)

            plt.figure(figsize=(10, 6))
            # plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_cv_score_list, '.r:', label='openness CV')

            # plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_cv_score_list, '.b:', label='conscientiousness CV')

            # plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_cv_score_list, '.g:', label='extraversion CV')

            # plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_cv_score_list, '.m:', label='agreeableness CV')

            # plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_cv_score_list, '.c:', label='neuroticism CV')

            max_open = max(max(self.openness_cv_score_list),
                           max(self.conscientiousness_cv_score_list),
                           max(self.extraversion_cv_score_list),
                           max(self.agreeableness_cv_score_list),
                           max(self.neuroticism_cv_score_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'traits CV accuracy vs. amount purchase threshold \n'
            title += ' Max: ' + str(round(max_open, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(cur_penalty)\
                     + ' Gap:' + str(self.h_limit) + '-' + str(self.l_limit)

            plt.title(title)
            plt.ylabel('Test accuracy')
            plt.xlabel('Threshold purchase amount')
            # plt.ylim(0.4, 1)
            # plot_name = cur_directory + 'logistic_C=' + str(cur_C) + '_penalty=' \
            #             + str(cur_penalty) + '_max=' + str(round(max_open, 2)) + '_gap=' + str(
            #     bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '.png'
            plot_name = str(round(max_open, 2)) + '_CV_Accuracy_k=' + str(k_best) + '_penalty=' + str(cur_penalty) + '_gap=' + str(
                self.h_limit) + '_' + str(self.l_limit) + '_max=' + str(round(max_open, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            logging.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass

        return

    # plot traits accuracy versus - logistic
    def plot_traits_roc_versus_threshold(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt

        try:
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 2, 1)

            plt.figure(figsize=(10, 6))
            plt.plot(self.threshold_list, self.openness_score_roc_list, '.r-', label='openness')
            # plt.plot(self.threshold_list, self.openness_cv_score_list, '.r:', label='openness CV')

            plt.plot(self.threshold_list, self.conscientiousness_score_roc_list, '.b-', label='conscientiousness')
            # plt.plot(self.threshold_list, self.conscientiousness_cv_score_list, '.b:', label='conscientiousness CV')

            plt.plot(self.threshold_list, self.extraversion_score_roc_list, '.g-', label='extraversion')
            # plt.plot(self.threshold_list, self.extraversion_cv_score_list, '.g:', label='extraversion CV')

            plt.plot(self.threshold_list, self.agreeableness_score_roc_list, '.m-', label='agreeableness')
            # plt.plot(self.threshold_list, self.agreeableness_cv_score_list, '.m:', label='agreeableness CV')

            plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            # plt.plot(self.threshold_list, self.neuroticism_cv_score_list, '.c:', label='neuroticism CV')

            max_open = max(max(self.openness_score_roc_list),
                           max(self.conscientiousness_score_roc_list),
                           max(self.extraversion_score_roc_list),
                           max(self.agreeableness_score_roc_list),
                           max(self.neuroticism_score_roc_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'traits test ROC score vs. amount purchase threshold \n'
            title += ' Max: ' + str(round(max_open, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(
                cur_penalty) \
                     + ' Gap:' + str(self.h_limit) + '-' + str(self.l_limit)

            plt.title(title)
            plt.ylabel('ROC Test score')
            plt.xlabel('Threshold purchase amount')
            # plt.ylim(0.4, 1)
            # plot_name = cur_directory + 'logistic_C=' + str(cur_C) + '_penalty=' \
            #             + str(cur_penalty) + '_max=' + str(round(max_open, 2)) + '_gap=' + str(
            #     bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '.png'
            plot_name = str(round(max_open, 2)) + '_ROC_k=' + str(k_best) + '_penalty=' + str(
                cur_penalty) + '_gap=' + str(
                self.h_limit) + '_' + str(self.l_limit) + '_max=' + str(round(max_open, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            logging.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass

        return

    # plot results - linear regression TODO
    def plot_traits_mae_versus_threshold_linear(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt
        try:

            plt.figure(figsize=(10, 6))
            # plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_score_mae_list, '.r:', label='openness mae')

            # plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_score_mae_list, '.b:', label='conscientiousness mae')

            # plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_score_mae_list, '.g:', label='extraversion mae')

            # plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_score_mae_list, '.m:', label='agreeableness mae')

            # plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_score_mae_list, '.c:', label='neuroticism mae')

            min_mae = min(min(self.openness_score_mae_list),
                          min(self.conscientiousness_score_mae_list),
                          min(self.extraversion_score_mae_list),
                          min(self.agreeableness_score_mae_list),
                          min(self.neuroticism_score_mae_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'personalty traits MAE vs. amount purchase threshold \n'
            title += ' Min MAE: ' + str(round(min_mae, 2)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(cur_penalty)


            plt.title(title)
            plt.ylabel('Test accuracy')
            plt.xlabel('Threshold purchase amount')
            plot_name = str(round(min_mae, 2)) + '_MAE_k=' + str(k_best) + '_penalty=' + str(cur_penalty) + '_min_mae=' + str(round(min_mae, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            logging.info('min MAE: ' + str(round(min_mae, 3)))
            logging.info('save plot: ' + str(plot_path))


        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass
        return

    # plot results - linear regression TODO
    def plot_traits_pearson_versus_threshold_linear(self, cur_penalty, k_best):

        import matplotlib.pyplot as plt
        try:

            plt.figure(figsize=(10, 6))
            # plt.plot(self.threshold_list, self.openness_score_list, '.r-', label='openness')
            plt.plot(self.threshold_list, self.openness_score_pearson_list, '.r:', label='openness pearson')

            # plt.plot(self.threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
            plt.plot(self.threshold_list, self.conscientiousness_score_pearson_list, '.b:', label='conscientiousness pearson')

            # plt.plot(self.threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
            plt.plot(self.threshold_list, self.extraversion_score_pearson_list, '.g:', label='extraversion pearson')

            # plt.plot(self.threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
            plt.plot(self.threshold_list, self.agreeableness_score_pearson_list, '.m:', label='agreeableness pearson')

            # plt.plot(self.threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')
            plt.plot(self.threshold_list, self.neuroticism_score_pearson_list, '.c:', label='neuroticism pearson')

            max_p = max(max(self.openness_score_pearson_list),
                          max(self.conscientiousness_score_pearson_list),
                          max(self.extraversion_score_pearson_list),
                          max(self.agreeableness_score_pearson_list),
                          max(self.neuroticism_score_pearson_list))

            plt.legend(loc='upper left')

            # plt.title('traits test accuracy vs. amount purchase threshold')
            title = 'personalty traits Pearson vs. amount purchase threshold \n'
            title += ' max Pearson: ' + str(round(max_p, 3)) + ' # features: ' + str(k_best) + ' Penalty: ' + str(
                cur_penalty)

            plt.title(title)
            plt.ylabel('Pearson correlation')
            plt.xlabel('Threshold purchase amount')
            plot_name = str(round(max_p, 3)) + '_Pearson_k=' + str(k_best) + '_penalty=' + str(
                cur_penalty) + '_max_pearson=' + str(round(max_p, 2)) + '.png'

            plot_path = self.plot_directory + plot_name
            plt.savefig(plot_path, bbox_inches='tight')
            # plt.show()
            plt.close()
            logging.info('max pearson: ' + str(round(max_p, 3)))
            logging.info('save plot: ' + str(plot_path))

        except Exception, e:
            print('found problem')
            print('Failed massage: ' + str(e))
            print(Exception)
            pass
        return


def main(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
         dir_logistic_results, dict_feature_flag, model_method, user_type, l_limit, h_limit, k_best_list,
         threshold_list, penalty, bool_slice_gap_percentile):

    wrapper_obj = Wrapper(participant_file, item_aspects_file, purchase_history_file, valid_users_file,
                          dir_analyze_name, dir_logistic_results, dict_feature_flag, model_method, user_type, l_limit,
                          h_limit, k_best_list, threshold_list, penalty, bool_slice_gap_percentile)

    wrapper_obj.init_debug_log()        # init debug once - log file
    wrapper_obj.run_models()

if __name__ == '__main__':

    # input file name
    # Hadas extraction
    # participant_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/Merge all/clean_participant_549_2018-03-15 15:31:43.csv'
    # item_aspects_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/Merge all/personality_item_aspects.csv'
    # purchase_history_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/Merge all/personality_purchase_history.csv'
    # valid_users_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/Merge all/personality_valid_users.csv'

    # Guy extraction
    participant_file = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/clean_participant_695_2018-05-13 16:54:12.csv'
    item_aspects_file = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/Item Aspects.csv'
    purchase_history_file = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/personality_purchase_history.csv'
    valid_users_file = '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/personality_valid_users.csv'

    dir_analyze_name = '/Users/gelad/Personality-based-commerce/BFI_results/analyze_CF/'
    dir_logistic_results = '/Users/gelad/Personality-based-commerce/BFI_results/'

    dict_feature_flag = {
        'time_purchase_ratio_feature_flag': True,
        'time_purchase_meta_feature_flag': True,
        'vertical_ratio_feature_flag': True,
        'purchase_percentile_feature_flag': True,
        'user_meta_feature_flag': True,
        'aspect_feature_flag': False
    }

    model_method = 'logistic'           # 'logistic'/'linear'
    user_type = 'all'                   # 'all'/'cf'/'ebay-tech'
    l_limit = 0.3                       # max percentile value for low group
    h_limit = 0.7                       #
    k_best_list = [5, 8, 12, 15]        # list for select K best features
    threshold_list = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    penalty = ['l1', 'l2']
    bool_slice_gap_percentile = True

    main(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name,
         dir_logistic_results, dict_feature_flag, model_method, user_type, l_limit, h_limit, k_best_list,
         threshold_list, penalty, bool_slice_gap_percentile)
