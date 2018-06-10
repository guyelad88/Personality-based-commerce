from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt


class CalculateScore:

    def __init__(self, file_name, dir_analyze_name):

        self.df = pd.DataFrame()
        self.file_name = file_name
        self.dir_analyze_name = dir_analyze_name
        self.avg_openness = 0
        self.avg_conscientiousness = 0
        self.avg_extraversion = 0
        self.avg_agreeableness = 0
        self.avg_neuroticism = 0

        self.ratio_hundred_openness = 0
        self.ratio_hundred_conscientiousness = 0
        self.ratio_hundred_extraversion = 0
        self.ratio_hundred_agreeableness = 0
        self.ratio_hundred_neuroticism = 0

        self.question_openness = [5, 10, 15, 20, 25, 30, 35, 40, 41, 44]
        self.question_conscientiousness = [3, 8, 13, 18, 23, 28, 33, 43]
        self.question_extraversion = [1, 6, 11, 16, 21, 26, 31, 36]
        self.question_agreeableness = [2, 7, 12, 17, 22, 27, 32, 37, 42]
        self.question_neuroticism = [4, 9, 14, 19, 24, 29, 34, 39]

        self.openness_score_list = list()
        self.conscientiousness_score_list = list()
        self.extraversion_score_list = list()
        self.agreeableness_score_list = list()
        self.neuroticism_score_list = list()

    # build log object
    def init_debug_log(self):
        import logging
        logging.basicConfig(filename='/Users/sguyelad/PycharmProjects/research/BFI_results/log/analyze_results.log',
                            filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info("")
        logging.info("")
        logging.info("start log program")

    # load csv and clean missing
    def load_clean_csv_results(self):

        self.df = pd.read_csv(self.file_name)

        self.df.rename(columns={'Email address': 'Username'}, inplace=True)
        prev_clean_row = self.df.shape[0]
        self.df.dropna(axis=0, how='any', inplace=True)
        after_clean_row = self.df.shape[0]

        logging.info('Number of deleted row: ' + str(prev_clean_row-after_clean_row))

    # save csv
    def save_data(self):
        self.df.reset_index(drop=True, inplace=True)
        save_file_name = self.file_name[:-4] + '_manipulate.csv'
        self.df.to_csv(save_file_name)
        return

    # reverse all relevant values
    def change_reverse_value(self):
        reverse_col = [2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43]
        for cur_rcol in reverse_col:
            start_str_cur = str(cur_rcol) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]
            # print(filter_col)
            logging.info('Change column values (reverse mode): ' + str(filter_col))
            self.df[filter_col] = self.df[filter_col].apply(lambda x: 6 - x)
        return

    # calculate average value
    def cal_participant_traits_values(self):

        # add empty columns
        self.df["openness_trait"] = np.nan
        self.df["conscientiousness_trait"] = np.nan
        self.df["extraversion_trait"] = np.nan
        self.df["agreeableness_trait"] = np.nan
        self.df["neuroticism_trait"] = np.nan
        self.df["age_group"] = ''  # a (15-24), b (25-29), c(30-34), d(35-39), e(40-100)

        # add average traits columns
        for (idx, row_participant) in self.df.iterrows():
            logging.info('Calculate traits for participant: ' + str(row_participant['Username']))
            participant_score = self.calculate_individual_score(idx, row_participant)

        for (idx, row_participant) in self.df.iterrows():
            # logging.info('Calculate percentile traits for participant: ' + str(row_participant['Email address']))
            self.cal_participant_traits_percentile_values(participant_score)
        return

    # calculate traits values for one participant
    def calculate_individual_score(self, idx, row_participant):

        participant_score = dict()
        op_trait = self.cal_participant_traits(row_participant, self.question_openness,
                                                     self.ratio_hundred_openness)

        self.df.set_value(idx, 'openness_trait', op_trait)
        self.openness_score_list.append(op_trait)
        participant_score['openness_trait'] = op_trait

        co_trait = self.cal_participant_traits(row_participant, self.question_conscientiousness,
                                                              self.ratio_hundred_conscientiousness)
        self.df.set_value(idx, 'conscientiousness_trait', co_trait)
        self.conscientiousness_score_list.append(co_trait)
        participant_score['conscientiousness_trait'] = co_trait

        ex_trait = self.cal_participant_traits(row_participant, self.question_extraversion,
                                                         self.ratio_hundred_extraversion)
        self.df.set_value(idx, 'extraversion_trait', ex_trait)
        self.extraversion_score_list.append(ex_trait)
        participant_score['extraversion_trait'] = ex_trait

        ag_trait = self.cal_participant_traits(row_participant, self.question_agreeableness,
                                                          self.ratio_hundred_agreeableness)
        self.df.set_value(idx, 'agreeableness_trait', ag_trait)
        self.agreeableness_score_list.append(ag_trait)
        participant_score['agreeableness_trait'] = ag_trait

        ne_trait = self.cal_participant_traits(row_participant, self.question_neuroticism,
                                                        self.ratio_hundred_neuroticism)
        self.df.set_value(idx, 'neuroticism_trait', ne_trait)
        self.neuroticism_score_list.append(ne_trait)
        participant_score['neuroticism_trait'] = ne_trait

        # a (15-24), b (25-29), c(30-34), d(35-39), e(40-100)
        if row_participant['Age'] <= 24:
            self.df.set_value(idx, 'age_group', 'a')
        elif row_participant['Age'] <= 29:
            self.df.set_value(idx, 'age_group', 'b')
        elif row_participant['Age'] <= 34:
            self.df.set_value(idx, 'age_group', 'c')
        elif row_participant['Age'] <= 39:
            self.df.set_value(idx, 'age_group', 'd')
        else:
            self.df.set_value(idx, 'age_group', 'e')
        return participant_score

    # calculate percentile value
    def cal_participant_traits_percentile_values(self, participant_score):

        participant_percentile = dict()
        participant_percentile['openness_trait'] = float(sum(i < participant_score['openness_trait'] for i in self.openness_score_list))/float(len(self.openness_score_list)-1)
        participant_percentile['conscientiousness_trait'] = float(sum(
            i < participant_score['conscientiousness_trait'] for i in self.conscientiousness_score_list))/float(len(self.conscientiousness_score_list)-1)
        participant_percentile['extraversion_trait'] = float(sum(
            i < participant_score['extraversion_trait'] for i in self.extraversion_score_list))/float(len(self.extraversion_score_list)-1)
        participant_percentile['agreeableness_trait'] = float(sum(
            i < participant_score['agreeableness_trait'] for i in self.agreeableness_score_list))/float(len(self.agreeableness_score_list)-1)
        participant_percentile['neuroticism_trait'] = float(sum(
            i < participant_score['neuroticism_trait'] for i in self.neuroticism_score_list))/float(len(self.neuroticism_score_list)-1)

        self.df["openness_trait"] = np.nan
        self.df["conscientiousness_trait"] = np.nan
        self.df["extraversion_trait"] = np.nan
        self.df["agreeableness_trait"] = np.nan
        self.df["neuroticism_trait"] = np.nan
        self.df["age_group"] = ''  # a (15-24), b (25-29), c(30-34), d(35-39), e(40-100)

        # add average traits columns
        for (idx, row_participant) in self.df.iterrows():
            logging.info('Calculate traits for participant: ' + str(row_participant['Username']))
            self.calculate_individual_score(idx, row_participant)

        return

    # calculate average traits value
    def cal_participant_traits(self, row, cur_trait_list, ratio):
        trait_val = 0
        for cur_col in cur_trait_list:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]   # find col name
            trait_val += row[filter_col]
        trait_val = float(trait_val)/float(len(cur_trait_list))     # mean of traits
        return trait_val

    # analyze data
    def analyze_data(self):
        self.compare_traits_site()
        self.compare_traits_gender()
        self.compare_traits_age()
        return

    ###### generate plots

    # compare traits per site
    def compare_traits_site(self):

        grouped = self.df.groupby(['Type'])
        for name, group in grouped:
            if name == 'eBay':
                eBay_label = 'eBay #p ' + str(group.shape[0])
            if name == 'Tech':
                tech_label = 'Tech #p ' + str(group.shape[0])

        avg_ebay = list()
        avg_tech = list()
        avg_ebay.append(round(self.df.groupby(['Type'])['openness_trait'].quantile([0.25])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['openness_trait'].quantile([0.25])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['conscientiousness_trait'].quantile([0.25])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['conscientiousness_trait'].quantile([0.25])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['extraversion_trait'].quantile([0.25])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['extraversion_trait'].quantile([0.25])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['agreeableness_trait'].quantile([0.25])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['agreeableness_trait'].quantile([0.25])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['neuroticism_trait'].quantile([0.25])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['neuroticism_trait'].quantile([0.25])['Tech'], 2))

        self.plot_compare(avg_ebay, avg_tech, eBay_label, tech_label,
                          ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                           'Neuroticism'], 'Q1_eBay_tech', 'Q1 Personality Traits vs. Site')


        avg_ebay = list()
        avg_tech = list()
        avg_ebay.append(round(self.df.groupby(['Type'])['openness_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['openness_trait'].median()['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['conscientiousness_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['conscientiousness_trait'].median()['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['extraversion_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['extraversion_trait'].median()['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['agreeableness_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['agreeableness_trait'].median()['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['neuroticism_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['neuroticism_trait'].median()['Tech'], 2))

        self.plot_compare(avg_ebay, avg_tech, eBay_label, tech_label,
                          ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                           'Neuroticism'], 'median_eBay_tech', 'Median Personality Traits vs. Site')

        avg_ebay = list()
        avg_tech = list()
        avg_ebay.append(round(self.df.groupby(['Type'])['openness_trait'].quantile([0.75])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['openness_trait'].quantile([0.75])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['conscientiousness_trait'].quantile([0.75])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['conscientiousness_trait'].quantile([0.75])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['extraversion_trait'].quantile([0.75])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['extraversion_trait'].quantile([0.75])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['agreeableness_trait'].quantile([0.75])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['agreeableness_trait'].quantile([0.75])['Tech'], 2))
        avg_ebay.append(round(self.df.groupby(['Type'])['neuroticism_trait'].quantile([0.75])['eBay'], 2))
        avg_tech.append(round(self.df.groupby(['Type'])['neuroticism_trait'].quantile([0.75])['Tech'], 2))

        self.plot_compare(avg_ebay, avg_tech, eBay_label, tech_label,
                          ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                           'Neuroticism'], 'Q3_eBay_tech', 'Q3 Personality Traits vs. Site')

        return

    # compare traits per site
    def compare_traits_gender(self):

        grouped = self.df.groupby(['Gender'])
        for name, group in grouped:
            if name == 'Female':
                female_label = 'Female #p ' + str(group.shape[0])
            if name == 'Male':
                male_label = 'Male #p ' + str(group.shape[0])

        avg_female = list()
        avg_male = list()
        avg_female.append(round(self.df.groupby(['Gender'])['openness_trait'].quantile([0.25])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['openness_trait'].quantile([0.25])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['conscientiousness_trait'].quantile([0.25])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['conscientiousness_trait'].quantile([0.25])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['extraversion_trait'].quantile([0.25])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['extraversion_trait'].quantile([0.25])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['agreeableness_trait'].quantile([0.25])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['agreeableness_trait'].quantile([0.25])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['neuroticism_trait'].quantile([0.25])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['neuroticism_trait'].quantile([0.25])['Male'], 2))

        self.plot_compare(avg_female, avg_male, female_label, male_label,
                          ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                           'Neuroticism'], 'Q1_Female_Male', 'Q1 Personality Traits vs. Gender')

        avg_female = list()
        avg_male = list()
        avg_female.append(round(self.df.groupby(['Gender'])['openness_trait'].median()['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['openness_trait'].median()['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['conscientiousness_trait'].median()['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['conscientiousness_trait'].median()['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['extraversion_trait'].median()['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['extraversion_trait'].median()['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['agreeableness_trait'].median()['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['agreeableness_trait'].median()['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['neuroticism_trait'].median()['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['neuroticism_trait'].median()['Male'], 2))

        self.plot_compare(avg_female, avg_male, female_label, male_label,
                                ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                                 'Neuroticism'], 'median_Female_Male', 'Median Personality Traits vs. Gender')

        avg_female = list()
        avg_male = list()
        avg_female.append(round(self.df.groupby(['Gender'])['openness_trait'].quantile([0.75])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['openness_trait'].quantile([0.75])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['conscientiousness_trait'].quantile([0.75])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['conscientiousness_trait'].quantile([0.75])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['extraversion_trait'].quantile([0.75])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['extraversion_trait'].quantile([0.75])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['agreeableness_trait'].quantile([0.75])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['agreeableness_trait'].quantile([0.75])['Male'], 2))
        avg_female.append(round(self.df.groupby(['Gender'])['neuroticism_trait'].quantile([0.75])['Female'], 2))
        avg_male.append(round(self.df.groupby(['Gender'])['neuroticism_trait'].quantile([0.75])['Male'], 2))

        self.plot_compare(avg_female, avg_male, female_label, male_label,
                          ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                           'Neuroticism'], 'Q3_Female_Male', 'Q3 Personality Traits vs. Gender')

        return

    # compare traits per age (percentile)
    def compare_traits_age(self):
        self.compare_traits_age_q1()
        self.compare_traits_age_median()
        self.compare_traits_age_q3()

    def compare_traits_age_q1(self):

        grouped = self.df.groupby(['age_group'])
        for name, group in grouped:
            if name == 'a':
                a_label = '-24 #p ' + str(group.shape[0])
            if name == 'b':
                b_label = '25-29 #p ' + str(group.shape[0])
            if name == 'c':
                c_label = '30-34 #p ' + str(group.shape[0])
            if name == 'd':
                d_label = '35-39 #p ' + str(group.shape[0])
            if name == 'e':
                e_label = '40- #p ' + str(group.shape[0])

        avg_a = list()
        avg_b = list()
        avg_c = list()
        avg_d = list()
        avg_e = list()

        avg_a.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.25])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.25])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.25])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.25])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.25])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.25])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.25])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.25])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.25])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.25])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.25])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.25])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.25])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.25])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.25])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.25])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.25])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.25])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.25])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.25])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.25])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.25])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.25])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.25])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.25])['e'], 2))

        self.plot_multi_compare(avg_a, avg_b, avg_c, avg_d, avg_e, a_label, b_label, c_label, d_label, e_label,
                                ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                                 'Neuroticism'], 'Q1_age_multi', 'Q1 Personality traits per age')
        return

    def compare_traits_age_median(self):

        grouped = self.df.groupby(['age_group'])
        for name, group in grouped:
            if name == 'a':
                a_label = '-24 #p ' + str(group.shape[0])
            if name == 'b':
                b_label = '25-29 #p ' + str(group.shape[0])
            if name == 'c':
                c_label = '30-34 #p ' + str(group.shape[0])
            if name == 'd':
                d_label = '35-39 #p ' + str(group.shape[0])
            if name == 'e':
                e_label = '40- #p ' + str(group.shape[0])

        avg_a = list()
        avg_b = list()
        avg_c = list()
        avg_d = list()
        avg_e = list()

        avg_a.append(round(self.df.groupby(['age_group'])['openness_trait'].median()['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['openness_trait'].median()['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['openness_trait'].median()['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['openness_trait'].median()['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['openness_trait'].median()['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].median()['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].median()['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].median()['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].median()['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].median()['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['extraversion_trait'].median()['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['extraversion_trait'].median()['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['extraversion_trait'].median()['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['extraversion_trait'].median()['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['extraversion_trait'].median()['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].median()['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].median()['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].median()['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].median()['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].median()['e'], 2))

        avg_a.append(self.df.groupby(['age_group'])['neuroticism_trait'].median()['a'])
        avg_b.append(self.df.groupby(['age_group'])['neuroticism_trait'].median()['b'])
        avg_c.append(self.df.groupby(['age_group'])['neuroticism_trait'].median()['c'])
        #grouped = self.df.groupby(['age_group'])
        #for name, group in grouped:
        #    print(name)
        #    #print(group)
        #    print(group['neuroticism_trait'].quantile([0.3, 0.5, 0.7]))
        #print(self.df.groupby(['age_group'])['neuroticism_trait'])
        #print(self.df.groupby(['age_group'])['neuroticism_trait'])
        #print(self.df.groupby(['age_group'])['neuroticism_trait'])
        avg_d.append(self.df.groupby(['age_group'])['neuroticism_trait'].median()['d'])
        avg_e.append(self.df.groupby(['age_group'])['neuroticism_trait'].median()['e'])

        self.plot_multi_compare(avg_a, avg_b, avg_c, avg_d, avg_e, a_label, b_label, c_label, d_label, e_label,
                          ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                           'Neuroticism'], 'median_age_multi', 'Median Personality traits per age')
        return

    def compare_traits_age_q3(self):

        grouped = self.df.groupby(['age_group'])
        for name, group in grouped:
            if name == 'a':
                a_label = '-24 #p ' + str(group.shape[0])
            if name == 'b':
                b_label = '25-29 #p ' + str(group.shape[0])
            if name == 'c':
                c_label = '30-34 #p ' + str(group.shape[0])
            if name == 'd':
                d_label = '35-39 #p ' + str(group.shape[0])
            if name == 'e':
                e_label = '40- #p ' + str(group.shape[0])

        avg_a = list()
        avg_b = list()
        avg_c = list()
        avg_d = list()
        avg_e = list()

        avg_a.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.75])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.75])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.75])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.75])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['openness_trait'].quantile([0.75])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.75])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.75])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.75])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.75])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['conscientiousness_trait'].quantile([0.75])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.75])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.75])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.75])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.75])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['extraversion_trait'].quantile([0.75])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.75])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.75])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.75])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.75])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['agreeableness_trait'].quantile([0.75])['e'], 2))

        avg_a.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.75])['a'], 2))
        avg_b.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.75])['b'], 2))
        avg_c.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.75])['c'], 2))
        avg_d.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.75])['d'], 2))
        avg_e.append(round(self.df.groupby(['age_group'])['neuroticism_trait'].quantile([0.75])['e'], 2))

        self.plot_multi_compare(avg_a, avg_b, avg_c, avg_d, avg_e, a_label, b_label, c_label, d_label, e_label,
                                ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                                 'Neuroticism'], 'Q3_age_multi', 'Q3 Personality traits per age')
        return

    # two columns compare
    def plot_multi_compare(self, first_col, sec_col, third_col, four_col, five_col, first_label, sec_label, third_label, four_label, five_label, xticks, plt_name, title):
        n_groups = len(xticks)
        fig, ax = plt.subplots(figsize=(9, 6))
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.65

        rects1 = plt.bar(index, first_col, bar_width,
                         alpha=opacity,
                         color='b',
                         label=first_label)

        rects2 = plt.bar(index + bar_width, sec_col, bar_width,
                         alpha=opacity,
                         color='g',
                         label=sec_label)

        rects3 = plt.bar(index + 2*bar_width, third_col, bar_width,
                         alpha=opacity,
                         color='r',
                         label=third_label)

        rects4 = plt.bar(index + 3 * bar_width, four_col, bar_width,
                         alpha=opacity,
                         color='c',
                         label=four_label)

        rects5 = plt.bar(index + 4 * bar_width, five_col, bar_width,
                         alpha=opacity,
                         color='m',
                         label=five_label)
        # label='Average eBay employees score')

        plt.xlabel('Personality traits')
        plt.ylabel('Scores')
        plt.title(title)
        plt.xticks(index + bar_width, xticks, rotation=20)
        axes = plt.gca()
        axes.set_ylim([
            min(min(first_col)-0.5, min(sec_col)-0.5, min(third_col)-0.5, min(four_col)-0.5, min(five_col)-0.5),
            max(max(first_col)+0.5, max(sec_col)+0.5, max(third_col)+0.5, max(four_col)+0.5, max(five_col)+0.5)
        ])
        ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), (first_label, sec_label, third_label, four_label, five_label))

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%.2f' % height,
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        autolabel(rects5)

        plt.tight_layout()
        # plt.show()
        plot_name = dir_analyze_name + plt_name + '.png'
        plt.savefig(plot_name, bbox_inches='tight')

        plt.close()

        return

    # two columns compare
    def plot_compare(self, first_col, sec_col, first_label, sec_label, xticks, plt_name, title):
        n_groups = len(xticks)
        fig, ax = plt.subplots(figsize=(9, 6))
        index = np.arange(n_groups)
        bar_width = 0.30
        opacity = 0.65

        rects1 = plt.bar(index, first_col, bar_width,
                         alpha=opacity,
                         color='b',
                         label=first_label)

        rects2 = plt.bar(index + bar_width, sec_col, bar_width,
                         alpha=opacity,
                         color='g',
                         label=sec_label)
        # label='Average eBay employees score')

        plt.xlabel('Personality traits')
        plt.ylabel('Scores')
        plt.title(title)
        plt.xticks(index + bar_width, xticks, rotation=20)
        axes = plt.gca()
        axes.set_ylim([min(min(first_col)-0.5, min(sec_col)-0.5), max(max(first_col)+0.5, max(sec_col)+0.5)])
        ax.legend((rects1[0], rects2[0]), (first_label, sec_label))

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%.2f' % height,
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        # plt.show()
        plot_name = dir_analyze_name + plt_name + '.png'
        plt.savefig(plot_name, bbox_inches='tight')

        plt.close()

        return

def main(file_name, dir_analyze_name):
    calculate_obj = CalculateScore(file_name, dir_analyze_name)            # create object and variables
    calculate_obj.init_debug_log()                  # init log file
    calculate_obj.load_clean_csv_results()          # load dataset
    calculate_obj.change_reverse_value()            # change specific column into reverse mode

    # calculate tratis values for participant
    calculate_obj.cal_participant_traits_values()
    #v calculate_obj.cal_participant_traits_percentile_values()

    calculate_obj.analyze_data()
    calculate_obj.save_data()                       # insert manipulate data into csv file


if __name__ == '__main__':
    file_name = '/Users/sguyelad/PycharmProjects/research/BFI_results/data/Personality test - 127 participant.csv'
    dir_analyze_name = '/Users/sguyelad/PycharmProjects/research/BFI_results/analyze/'
    main(file_name, dir_analyze_name)