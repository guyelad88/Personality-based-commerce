from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from build_feature_dataset import CalculateScore

class Wrapper:

    def __init__(self, participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name):

        # file arguments
        self.participant_file = participant_file
        self.item_aspects_file = item_aspects_file
        self.purchase_history_file = purchase_history_file
        self.valid_users_file = valid_users_file
        self.dir_analyze_name = dir_analyze_name

        self.openness_score_list = list()
        self.conscientiousness_score_list = list()
        self.extraversion_score_list = list()
        self.agreeableness_score_list = list()
        self.neuroticism_score_list = list()

        self.max_score = {
            "openness": {'i': -1, 'score': 0.0},
            "agreeableness": {'i': -1, 'score': 0.0},
            "conscientiousness": {'i': -1, 'score': 0.0},
            "extraversion": {'i': -1, 'score': 0.0},
            "neuroticism": {'i': -1, 'score': 0.0},
        }

        self.i = int

    def run_main(self):
        for i in range(0, 15):
            self.i = i
            print('')
            print('Iteration number: ' + str(i+1))
            self.run_experiements()
        self.summarize()
        return

    def run_experiements(self):

        threshold_list = [0, 10, 20, 30, 50, 60]
        C = [1, 2, 3, 4.5]
        penalty = ['l1', 'l2']

        C = [4.5]
        penalty = ['l2']
        threshold_list =[20]
        bool_slice_gap_percentile = False
        bool_normalize_features = True

        self.time_purchase_ratio_feature_flag = True
        self.time_purchase_meta_feature_flag = True
        self.vertical_ratio_feature_flag = True
        self.purchase_percentile_feature_flag = True
        self.user_meta_feature_flag = True

        bool_random = False
        if bool_random:
            import random
            time_a = random.uniform(0, 1)
            time_b = random.uniform(0, 1)
            ver = random.uniform(0, 1)
            purchase = random.uniform(0, 1)
            meta = random.uniform(0, 1)

            if time_a > 0.4:
                self.time_purchase_ratio_feature_flag = True
            else:
                self.time_purchase_ratio_feature_flag = False

            if time_b > 0.4:
                self.time_purchase_meta_feature_flag = True
            else:
                self.time_purchase_meta_feature_flag = False

            if ver > 0.4:
                self.vertical_ratio_feature_flag = True
            else:
                self.vertical_ratio_feature_flag = False

            if purchase > 0.4:
                self.purchase_percentile_feature_flag = True
            else:
                self.purchase_percentile_feature_flag = False

            if meta > 0.4:
                self.user_meta_feature_flag = True
            else:
                self.user_meta_feature_flag = False

            if max(time_a, time_b, ver, purchase, meta) < 0.4:
                return

        import os
        from time import gmtime, strftime
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        cur_directory = self.dir_analyze_name + '/logistic_' + str(cur_time) + '_gap=' + str(bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '/'
        self.cur_directory = cur_directory

        if not os.path.exists(cur_directory):
            os.makedirs(cur_directory)

        # dict for storing best results
        self.openness_score_dict = dict()
        self.conscientiousness_score_dict = dict()
        self.extraversion_score_dict = dict()
        self.agreeableness_score_dict = dict()
        self.neuroticism_score_dict = dict()

        for cur_C in C:
            for cur_penalty in penalty:
                print('C: ' + str(cur_C) + ' Penalty: ' + str(cur_penalty))
                self.openness_score_list = list()
                self.conscientiousness_score_list = list()
                self.extraversion_score_list = list()
                self.agreeableness_score_list = list()
                self.neuroticism_score_list = list()

                for threshold_purchase in threshold_list:

                    calculate_obj = CalculateScore(participant_file, item_aspects_file, purchase_history_file, valid_users_file,
                                                   dir_analyze_name, threshold_purchase,bool_slice_gap_percentile,
                                                   bool_normalize_features, cur_C, cur_penalty, self.time_purchase_ratio_feature_flag,
                                                   self.time_purchase_meta_feature_flag, self.vertical_ratio_feature_flag,
                                                   self.purchase_percentile_feature_flag, self.user_meta_feature_flag)

                    calculate_obj.init_debug_log()          # init log file
                    calculate_obj.load_clean_csv_results()  # load data set
                    calculate_obj.clean_df()                # clean df - e.g. remain valid users only
                    calculate_obj.create_feature_list()     # create x_feature

                    # calculate personality trait per user + percentile per trait
                    calculate_obj.change_reverse_value()            # change specific column into reverse mode
                    calculate_obj.cal_participant_traits_values()   # calculate average traits and percentile value
                    calculate_obj.insert_gender_feature()           # add gender feature

                    calculate_obj.extract_user_purchase_connection()    # insert purchase and vertical type to model
                    calculate_obj.insert_money_feature()                # add feature contain money issue
                    calculate_obj.insert_time_feature()                 # add time purchase feature

                    calculate_obj.calculate_logistic_regression()         # predict traits H or L
                    # calculate_obj.calculate_linear_regression()             # predict traits score

                    print(threshold_purchase)
                    cur_key = 'C_' + str(cur_C) + '_Penalty_' + str(cur_penalty) + '_Threshold_' + str(threshold_purchase)
                    self.openness_score_dict[cur_key] = calculate_obj.logistic_regression_accuracy['openness']
                    self.conscientiousness_score_dict[cur_key] = calculate_obj.logistic_regression_accuracy['conscientiousness']
                    self.extraversion_score_dict[cur_key] = calculate_obj.logistic_regression_accuracy['extraversion']
                    self.agreeableness_score_dict[cur_key] = calculate_obj.logistic_regression_accuracy['agreeableness']
                    self.neuroticism_score_dict[cur_key] = calculate_obj.logistic_regression_accuracy['neuroticism']

                    # print(calculate_obj.logistic_regression_accuracy)
                    self.openness_score_list.append(calculate_obj.logistic_regression_accuracy['openness'])
                    self.conscientiousness_score_list.append(calculate_obj.logistic_regression_accuracy['conscientiousness'])
                    self.extraversion_score_list.append(calculate_obj.logistic_regression_accuracy['extraversion'])
                    self.agreeableness_score_list.append(calculate_obj.logistic_regression_accuracy['agreeableness'])
                    self.neuroticism_score_list.append(calculate_obj.logistic_regression_accuracy['neuroticism'])

                # plot results
                import matplotlib.pyplot as plt
                try:
                    plt.plot(threshold_list, self.openness_score_list, '.r-', label='openness')
                    plt.plot(threshold_list, self.conscientiousness_score_list, '.b-', label='conscientiousness')
                    plt.plot(threshold_list, self.extraversion_score_list, '.g-', label='extraversion')
                    plt.plot(threshold_list, self.agreeableness_score_list, '.m-', label='agreeableness')
                    plt.plot(threshold_list, self.neuroticism_score_list, '.c-', label='neuroticism')

                    max_open = max(max(self.openness_score_list), max(self.conscientiousness_score_list), max(self.agreeableness_score_list))
                    plt.legend(loc='upper left')
                    plt.title('traits test accuracy vs. amount purchase threshold')
                    plt.ylabel('Test accuracy')
                    plt.xlabel('Threshold purchase amount')
                    plot_name = cur_directory + 'logistic_C=' + str(cur_C) +'_penalty=' \
                                + str(cur_penalty) + '_max=' + str(round(max_open,2)) + '_gap=' + str(bool_slice_gap_percentile) + '_norm=' + str(bool_normalize_features) + '.png'
                    plt.savefig(plot_name, bbox_inches='tight')
                    # plt.show()
                    plt.close()
                    print('plot save')
                except Exception:
                    print('found problem')
                    print(Exception)
                    pass

        self.calculate_best_combination_per_trait()

        return

    def calculate_best_combination_per_trait(self):
        import operator
        openness = sorted(self.openness_score_dict.items(), key=operator.itemgetter(1))
        openness.reverse()
        openness = openness[:5]
        conscientiousness = sorted(self.conscientiousness_score_dict.items(), key=operator.itemgetter(1))
        conscientiousness.reverse()
        conscientiousness = conscientiousness[:5]
        extraversion = sorted(self.extraversion_score_dict.items(), key=operator.itemgetter(1))
        extraversion.reverse()
        extraversion = extraversion[:5]
        agreeableness = sorted(self.agreeableness_score_dict.items(), key=operator.itemgetter(1))
        agreeableness.reverse()
        agreeableness = agreeableness[:5]
        neuroticism = sorted(self.neuroticism_score_dict.items(), key=operator.itemgetter(1))
        neuroticism.reverse()
        neuroticism = neuroticism[:5]

        if self.max_score['openness']['score'] < openness[0][1]:
            self.max_score['openness']['score'] = openness[0][1]
            self.max_score['openness']['i'] = self.i
            self.max_score['openness']['conf'] = openness[0][0]

        if self.max_score['agreeableness']['score'] < agreeableness[0][1]:
            self.max_score['agreeableness']['score'] = agreeableness[0][1]
            self.max_score['agreeableness']['i'] = self.i
            self.max_score['agreeableness']['conf'] = agreeableness[0][0]

        if self.max_score['conscientiousness']['score'] < conscientiousness[0][1]:
            self.max_score['conscientiousness']['score'] = conscientiousness[0][1]
            self.max_score['conscientiousness']['i'] = self.i
            self.max_score['conscientiousness']['conf'] = conscientiousness[0][0]

        if self.max_score['extraversion']['score'] < extraversion[0][1]:
            self.max_score['extraversion']['score'] = extraversion[0][1]
            self.max_score['extraversion']['i'] = self.i
            self.max_score['extraversion']['conf'] = extraversion[0][0]

        if self.max_score['neuroticism']['score'] < neuroticism[0][1]:
            self.max_score['neuroticism']['score'] = neuroticism[0][1]
            self.max_score['neuroticism']['i'] = self.i
            self.max_score['neuroticism']['conf'] = neuroticism[0][0]

        text_file = open(self.cur_directory + 'summary.txt', "w")
        text_file.write('Iteration number: ' + str(self.i))
        text_file.write('time_purchase_ratio_feature_flag: ' + str(self.time_purchase_ratio_feature_flag))
        text_file.write('\n')
        text_file.write('time_purchase_meta_feature_flag: ' + str(self.time_purchase_meta_feature_flag))
        text_file.write('\n')
        text_file.write('vertical_ratio_feature_flag: ' + str(self.vertical_ratio_feature_flag))
        text_file.write('\n')
        text_file.write('purchase_percentile_feature_flag: ' + str(self.purchase_percentile_feature_flag))
        text_file.write('\n')
        text_file.write('user_meta_feature_flag: ' + str(self.user_meta_feature_flag))
        text_file.write('\n')
        text_file.write("openness")
        text_file.write(str(openness))
        text_file.write('\n')
        text_file.write('\n')
        text_file.write("agreeableness")
        text_file.write(str(agreeableness))
        text_file.write('\n')
        text_file.write('\n')
        text_file.write("conscientiousness")
        text_file.write(str(conscientiousness))
        text_file.write('\n')
        text_file.write('\n')
        text_file.write("extraversion")
        text_file.write(str(extraversion))
        text_file.write('\n')
        text_file.write('\n')
        text_file.write("neuroticism")
        text_file.write(str(neuroticism))
        text_file.close()

        print('openness')
        print(openness)
        print('agreeableness')
        print(agreeableness)
        print('conscientiousness')
        print(conscientiousness)
        print('extraversion')
        print(extraversion)
        print('neuroticism')
        print(neuroticism)

        return

    def summarize(self):
        import os
        from time import gmtime, strftime
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        cur_dir = self.dir_analyze_name + '/logistic_sum' + str(cur_time) + '/'
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        text_file = open(cur_dir + 'summary.txt', "w")
        for key, val in self.max_score.iteritems():
            text_file.write(str(key))
            text_file.write('\n')
            text_file.write("max iteration: " + str(val['i']))
            text_file.write('\n')
            text_file.write("max score: " + str(val['score']))
            text_file.write('\n')
            text_file.write("max configure: " + str(val['conf']))
            text_file.write('\n')
            text_file.write('\n')
            text_file.write('\n')
        text_file.close()
        return

def main(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name):

    wrapper_obj = Wrapper(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name)            # create object and variables
    # wrapper_obj.run_main()
    wrapper_obj.run_experiements()

if __name__ == '__main__':

    # input file name
    participant_file = '/Users/sguyelad/PycharmProjects/research/data/analyze_data/personality_139_participant.csv'
    item_aspects_file = '/Users/sguyelad/PycharmProjects/research/data/analyze_data/personality_item_aspects.csv'
    purchase_history_file = '/Users/sguyelad/PycharmProjects/research/data/analyze_data/personality_purchase_history.csv'
    valid_users_file = '/Users/sguyelad/PycharmProjects/research/data/analyze_data/personality_valid_users.csv'
    dir_analyze_name = '/Users/sguyelad/PycharmProjects/research/BFI_results/analyze_pic/'


    main(participant_file, item_aspects_file, purchase_history_file, valid_users_file, dir_analyze_name)
