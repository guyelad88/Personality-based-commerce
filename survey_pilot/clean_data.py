from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

class CleanData:
    '''
    remain only valid users
    check duplication
    remove according to threshold defined
    '''

    def __init__(self, participant_file, valid_users_file, threshold, len_user_name_threshold, duplication_method,
                 dir_save_results):

        # file arguments
        self.participant_file = participant_file
        self.valid_users_file = valid_users_file
        self.threshold = threshold                              # max gap in duplication
        self.len_user_name_threshold = len_user_name_threshold  # min valid len size
        self.duplication_method = duplication_method            # 'avg' / 'first'

        self.dir_save_results = dir_save_results            # save results of clean and valid particpants

        self.verbose_flag = True
        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # define data frame needed for analyzing data
        self.participant_df = pd.DataFrame()
        self.valid_users_df = pd.DataFrame()
        self.valid_participant_data = pd.DataFrame()        # contain only real users
        self.clean_participant_data = pd.DataFrame()        # after clean duplication, using threshold

        self.dup_gap_list = list()          # duplication with their max gap

        self.traits_list = [
            'agreeableness',
            'extraversion',
            'openness',
            'conscientiousness',
            'neuroticism'
        ]

    def init_debug_log(self):
        import logging

        lod_file_name = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/survey_pilot/log/' + 'clean_data_' + str(self.cur_time) + '.log'

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
    def load_clean_csv_results(self):

        self.participant_df = pd.read_csv(self.participant_file)
        self.valid_users_df = pd.read_csv(self.valid_users_file)

        if 'eBay site user name' not in list(self.valid_users_df):
            try:
                self.valid_users_df.rename(columns={'USER_SLCTD_ID': 'eBay site user name'}, inplace=True)
            except:
                pass

        return

    # delete users with un-valid user name
    def clean_df(self):
        '''
        remain only valid users
        :return:
        '''

        # real_user_name_list = list(self.valid_users_df['USER_SLCTD_ID'])
        real_user_name_list = list(self.valid_users_df['eBay site user name'])

        col_name = list(self.participant_df)
        self.valid_participant_data = pd.DataFrame(columns=col_name)

        for (idx, row_participant) in self.participant_df.iterrows():
            # print(row_participant['eBay site user name'])
            try:
                lower_first_name = row_participant['eBay site user name'].lower()
                # lower_first_name = row_participant['USER_SLCTD_ID'].lower()
                # self.participant_df.set_value(idx, 'eBay site user name', lower_first_name)
                if lower_first_name in real_user_name_list:
                    self.valid_participant_data = self.valid_participant_data.append(row_participant)

            except Exception:
                pass

        return

    # handler of duplication users - who to delete and keep
    def investigate_duplication(self):

        dup_above_gap = 0
        dup_below_gap = 0
        no_dup = 0
        cnt_len_below_threshold = 0

        col_name = ['Site', 'Email address', 'Full Name', 'Gender', 'eBay site user name', 'Nationality', 'Age',
                    'openness_trait', 'conscientiousness_trait', 'extraversion_trait', 'agreeableness_trait',
                    'neuroticism_trait']

        self.clean_participant_data = pd.DataFrame(columns=col_name)

        self.valid_participant_data['eBay site user name'].str.lower()
        ebay_user_group = self.valid_participant_data.groupby(['eBay site user name'])

        # iterate over each user and check max gap in duplication
        for ebay_user_name, user_group in ebay_user_group:

            if len(ebay_user_name) < self.len_user_name_threshold:
                logging.info('User name length below threshold: ' + str(ebay_user_name) + ' : ' + str(len(ebay_user_name)))
                cnt_len_below_threshold += 1
                continue

            if user_group.shape[0] > 1:

                diff_dup = {
                    'agreeableness': list(),
                    'extraversion': list(),
                    'openness': list(),
                    'conscientiousness': list(),
                    'neuroticism': list()
                }

                for (idx, row_participant) in user_group.iterrows():
                    for trait in self.traits_list:
                        cur_f = trait + '_trait'
                        diff_dup[trait].append(row_participant[cur_f])

                max_gap = 0
                for c_t, v_l in diff_dup.iteritems():
                    if max_gap < max(v_l)-min(v_l):
                        max_gap = max(v_l)-min(v_l)
                self.dup_gap_list.append(max_gap)

                # check max gap - decide whether to insert into clean_df or not
                if max_gap > threshold:
                    logging.info('Dup user above threshold: ' + str(ebay_user_name) + ' : ' + str(max_gap))
                    dup_above_gap += 1
                    continue

                else:
                    logging.info('Dup user below threshold: ' + str(ebay_user_name) + ' : ' + str(max_gap))
                    dup_below_gap += 1

                # insert duplication users (average his traits)
                if self.duplication_method == 'avg':
                    first_in_group = user_group.head(1)
                    first_in_group = first_in_group[['Site', 'Email address', 'Full Name', 'Gender', 'eBay site user name', 'Nationality', 'Age']]
                    for trait in self.traits_list:
                        cur_f = trait + '_trait'
                        first_in_group[cur_f] = sum(diff_dup[trait])/len(diff_dup[trait])

                    self.clean_participant_data = self.clean_participant_data.append(first_in_group)
            else:
                no_dup += 1
                logging.info('Unique user: ' + str(ebay_user_name))
                first_in_group = user_group.head(1)
                first_in_group = first_in_group[col_name]
                self.clean_participant_data = self.clean_participant_data.append(first_in_group)

        logging.info('Clean samples: ' + str(no_dup + dup_below_gap))
        logging.info('Duplication users valid: ' + str(dup_below_gap))
        logging.info('Duplication users not valid: ' + str(dup_above_gap))
        logging.info('Duplication users valid ratio: ' + str(float(dup_below_gap)/float(dup_below_gap+dup_above_gap)))
        logging.info('One users valid: ' + str(no_dup))
        logging.info('Users with user name not valid (below length threshold): ' + str(cnt_len_below_threshold))

        self.clean_participant_data.to_csv(self.dir_save_results + 'clean_participant_' +
                                           str(self.clean_participant_data.shape[0]) + '_' + str(self.cur_time) +
                                           '.csv')

        return

    # visualization of results
    def visualization_results(self):

        self.visu_trait_site()
        self.visu_max_gap()

    def visu_max_gap(self):

        plt.figure()

        plt.hist(self.dup_gap_list, bins=15)
        plt.ylabel('Amount')
        plt.xlabel('Max gap')
        plt.title('Histogram - Maximum gap in trait for duplication users')
        # plt.show()
        plot_name = self.dir_save_results + 'histogram_max_gap' + '.png'
        plt.savefig(plot_name)
        logging.info("Save histogram: " + str(plot_name))
        return

    def visu_trait_site(self):
        '''
        compare between site results - eBay Tech CF
        :return:
        '''

        if 'site' not in list(self.clean_participant_data):
            try:
                self.clean_participant_data.rename(columns={'Site': 'site'}, inplace=True)
            except:
                pass
        grouped = self.clean_participant_data.groupby(['site'])
        for name, group in grouped:
            if name == 'eBay':
                eBay_label = 'eBay #p ' + str(group.shape[0])
            if name == 'Technion':
                tech_label = 'Tech #p ' + str(group.shape[0])
            if name == 'CF':
                CF_label = 'CF #p ' + str(group.shape[0])

        avg_ebay = list()
        avg_tech = list()
        avg_CF = list()
        avg_ebay.append(round(self.clean_participant_data.groupby(['site'])['openness_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.clean_participant_data.groupby(['site'])['openness_trait'].median()['Technion'], 2))
        avg_CF.append(round(self.clean_participant_data.groupby(['site'])['openness_trait'].median()['CF'], 2))
        avg_ebay.append(round(self.clean_participant_data.groupby(['site'])['conscientiousness_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.clean_participant_data.groupby(['site'])['conscientiousness_trait'].median()['Technion'], 2))
        avg_CF.append(round(self.clean_participant_data.groupby(['site'])['conscientiousness_trait'].median()['CF'], 2))
        avg_ebay.append(round(self.clean_participant_data.groupby(['site'])['extraversion_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.clean_participant_data.groupby(['site'])['extraversion_trait'].median()['Technion'], 2))
        avg_CF.append(round(self.clean_participant_data.groupby(['site'])['extraversion_trait'].median()['CF'], 2))
        avg_ebay.append(round(self.clean_participant_data.groupby(['site'])['agreeableness_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.clean_participant_data.groupby(['site'])['agreeableness_trait'].median()['Technion'], 2))
        avg_CF.append(round(self.clean_participant_data.groupby(['site'])['agreeableness_trait'].median()['CF'], 2))
        avg_ebay.append(round(self.clean_participant_data.groupby(['site'])['neuroticism_trait'].median()['eBay'], 2))
        avg_tech.append(round(self.clean_participant_data.groupby(['site'])['neuroticism_trait'].median()['Technion'], 2))
        avg_CF.append(round(self.clean_participant_data.groupby(['site'])['neuroticism_trait'].median()['CF'], 2))

        self.plot_compare(avg_ebay, avg_tech, avg_CF, eBay_label, tech_label, CF_label,
                          ['Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                           'Neuroticism'], 'median_eBay_tech_CF', 'Median Personality Traits vs. Site')
        return

        # two columns compare

    # 3-bar plot - allow compare between sites
    def plot_compare(self, first_col, sec_col, third_col, first_label, sec_label, third_label, xticks, plt_name, title):
        n_groups = len(xticks)
        fig, ax = plt.subplots(figsize=(9, 6))
        index = np.arange(n_groups)
        bar_width = 0.20
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


        plt.xlabel('Personality traits')
        plt.ylabel('Scores')
        plt.title(title)
        plt.xticks(index + bar_width, xticks, rotation=20)
        axes = plt.gca()
        axes.set_ylim(
            [min(min(first_col) - 0.5, min(sec_col) - 0.5, min(third_col) - 0.5),
             max(max(first_col) + 0.5, max(sec_col) + 0.5,  max(third_col) + 0.5)])
        ax.legend((rects1[0], rects2[0], rects3[0]), (first_label, sec_label, third_label))

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

        plt.tight_layout()
        # plt.show()
        plot_name = self.dir_save_results + plt_name + '.png'
        plt.savefig(plot_name, bbox_inches='tight')

        plt.close()

        return


def main(participant_file, valid_users_file, threshold, len_user_name_threshold, duplication_method, dir_save_results):

    clean_data_obj = CleanData(participant_file, valid_users_file, threshold, len_user_name_threshold,
                               duplication_method, dir_save_results)

    clean_data_obj.init_debug_log()                      # init log file
    clean_data_obj.load_clean_csv_results()              # load data set
    clean_data_obj.clean_df()                            # clean df - e.g. remain valid users only
    clean_data_obj.investigate_duplication()             # remove users with dup + no real user name
    clean_data_obj.visualization_results()

if __name__ == '__main__':

    # input file name
    # participant_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/Merge all/merge_df_crowdflower_1057.csv'
    # valid_users_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/Merge all/personality_valid_users.csv'
    # dir_save_results = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/Merge all/'

    # participant_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/1425 users input/personality_participant_all_include_1287_CF total_1425.csv'
    participant_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/1425 users input/merge_df_crowdflower_1425.csv'
    valid_users_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/1425 users input/personality_valid_users.csv'
    dir_save_results = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/participant_data/1425 users input/'
    threshold = 0.5
    duplication_method = 'avg'  # 'avg', 'first'
    len_user_name_threshold = 4           #
    main(participant_file, valid_users_file, threshold, len_user_name_threshold, duplication_method, dir_save_results)