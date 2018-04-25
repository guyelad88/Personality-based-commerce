from __future__ import print_function
import os
import sys
import argparse
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt


class CalculateScore:

    def __init__(self):
        self.df = pd.DataFrame()

        self.avg_openness = 0
        self.avg_conscientiousness = 0
        self.avg_extraversion = 0
        self.avg_agreeableness = 0
        self.avg_neuroticism = 0

        self.prev_avg_openness = 3.19
        self.prev_avg_conscientiousness = 3.74
        self.prev_avg_extraversion = 3.35
        self.prev_avg_agreeableness = 3.37
        self.prev_avg_neuroticism = 2.71

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
        logging.basicConfig(filename='/Users/sguyelad/PycharmProjects/research/survey_pilot/log/log.log',
                            filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info("")
        logging.info("")
        logging.info("start log program")

    # load csv and clean missing
    def load_clean_csv_results(self):

        # self.df = pd.read_csv('/Users/sguyelad/PycharmProjects/research/survey_pilot/data/Personality test (BFI) - Technion 80 participant.csv')    # 14.1.18 09:00 - technion

        self.df = pd.read_csv('/Users/sguyelad/PycharmProjects/research/survey_pilot/data/Personality test - 67 participant.csv')  # 14.1.18 09:00

        '''if 'eBay email address' in list(self.df) and 'eBay email adress' in list(self.df):
            self.df['user email'] = self.df[['Username', 'eBay email address', 'eBay email adress']].apply(lambda x: ''.join(x), axis=1)
            del self.df['Username']
            del self.df['eBay email address']
            del self.df['eBay email adress']
            del self.df['eBay site user name']
        if 'eBay email address' in list(self.df):
            self.df['user email'] = self.df[['Username', 'eBay email address']].apply(lambda x: ''.join(x), axis=1)
            del self.df['Username']
            del self.df['eBay email address']
            del self.df['eBay site user name']
        else:
            a=5'''
        # self.df.rename(columns={'eBay email address': 'Username'}, inplace=True)
        self.df.rename(columns={'Email address': 'Username'}, inplace=True)
            #self.df['user email'] = self.df[['Username']].apply(lambda x: ''.join(x), axis=1)
        #self.df.to_csv('/Users/sguyelad/PycharmProjects/research/survey_pilot/data/elad_personality_check.csv', sep='\t', encoding='utf-8')
        #raise
        prev_clean_row = self.df.shape[0]
        self.df.dropna(axis=0, how='any', inplace=True)
        after_clean_row = self.df.shape[0]
        logging.info('Number of deleted row: ' + str(prev_clean_row-after_clean_row))
        # print(self.df)

    # reverse all relevant values
    def change_reverse_value(self):
        reverse_col = [2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43]
        for cur_rcol in reverse_col:
            start_str_cur = str(cur_rcol) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]
            # print(filter_col)
            logging.info('Change column values (reverse mode): ' + str(filter_col))
            self.df[filter_col] = self.df[filter_col].apply(lambda x: 6-x)
        return

    # calculate average score for the big five traits
    # openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism,
    # for each trait calculate mean score, ration to 50 to normalize output score later
    #
    # self.avg_trait -> mean for all traits (after reverse)
    # self.ratio_hundred_trait -> ratio regards to
    # self.ebay_avg_trait -> ratio (prior regards to article results) * dataset avg (eBay)
    def calculate_average_score(self):

        for cur_col in self.question_openness:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]
            self.avg_openness += self.df[filter_col].mean()
        self.avg_openness = self.avg_openness/len(self.question_openness)
        self.ratio_hundred_openness = float(50)/self.prev_avg_openness      # by article results
        logging.info('avg_openness : ' + str(self.avg_openness))

        self.ebay_avg_openness = self.avg_openness * self.ratio_hundred_openness
        logging.info('ebay_avg_openness : ' + str(self.ebay_avg_openness))

        for cur_col in self.question_conscientiousness:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]
            self.avg_conscientiousness += self.df[filter_col].mean()
        self.avg_conscientiousness = self.avg_conscientiousness/len(self.question_conscientiousness)
        self.ratio_hundred_conscientiousness = float(50) / self.prev_avg_conscientiousness
        logging.info('avg_conscientiousness : ' + str(self.avg_conscientiousness))

        self.ebay_avg_conscientiousness = self.avg_conscientiousness * self.ratio_hundred_conscientiousness
        logging.info('ebay_avg_conscientiousness : ' + str(self.ebay_avg_conscientiousness))

        for cur_col in self.question_extraversion:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]
            self.avg_extraversion += self.df[filter_col].mean()
        self.avg_extraversion = self.avg_extraversion/len(self.question_extraversion)
        self.ratio_hundred_extraversion = float(50) / self.prev_avg_extraversion
        logging.info('avg_extraversion : ' + str(self.avg_extraversion))

        self.ebay_avg_extraversion = self.avg_extraversion * self.ratio_hundred_extraversion
        logging.info('ebay_avg_extraversion : ' + str(self.ebay_avg_extraversion))

        for cur_col in self.question_agreeableness:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]
            self.avg_agreeableness += self.df[filter_col].mean()
        self.avg_agreeableness = self.avg_agreeableness/len(self.question_agreeableness)
        self.ratio_hundred_agreeableness = float(50) / self.prev_avg_agreeableness
        logging.info('avg_agreeableness : ' + str(self.avg_agreeableness))

        self.ebay_avg_agreeableness = self.avg_agreeableness * self.ratio_hundred_agreeableness
        logging.info('ebay_avg_agreeableness : ' + str(self.ebay_avg_agreeableness))

        for cur_col in self.question_neuroticism:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]
            self.avg_neuroticism += self.df[filter_col].mean()
        self.avg_neuroticism = self.avg_neuroticism/len(self.question_neuroticism)
        self.ratio_hundred_neuroticism = float(50) / self.prev_avg_neuroticism
        logging.info('avg_neuroticism : ' + str(self.avg_neuroticism))

        self.ebay_avg_neuroticism = self.avg_neuroticism * self.ratio_hundred_neuroticism
        logging.info('ebay_avg_neuroticism : ' + str(self.ebay_avg_neuroticism))
        return

    #
    def calculate_good_score(self):
        return

    # calculate traits values for one participant
    def calculate_individual_score(self, row_participant, check_percentile=False):

        participant_score = dict()
        participant_percentile = dict()
        participant_score['openness_trait'] = self.cal_participant_traits(row_participant, self.question_openness,
                                                     self.ratio_hundred_openness)
        participant_score[
            'conscientiousness_trait'] = self.cal_participant_traits(row_participant, self.question_conscientiousness,
                                                              self.ratio_hundred_conscientiousness)
        participant_score['extraversion_trait'] = self.cal_participant_traits(row_participant, self.question_extraversion,
                                                         self.ratio_hundred_extraversion)
        participant_score[
            'agreeableness_trait'] = self.cal_participant_traits(row_participant, self.question_agreeableness,
                                                          self.ratio_hundred_agreeableness)
        participant_score['neuroticism_trait'] = self.cal_participant_traits(row_participant, self.question_neuroticism,
                                                        self.ratio_hundred_neuroticism)

        logging.info('')
        logging.info('user name : ' + str(row_participant['Username']))
        logging.info('openness traits : ' + str(round(participant_score['openness_trait'], 2)))
        logging.info('conscientiousness traits : ' + str(round(participant_score['conscientiousness_trait'], 2)))
        logging.info('extraversion traits : ' + str(round(participant_score['extraversion_trait'], 2)))
        logging.info('agreeableness traits : ' + str(round(participant_score['agreeableness_trait'], 2)))
        logging.info('neuroticism traits : ' + str(round(participant_score['neuroticism_trait'], 2)))

        if check_percentile:
            participant_percentile['openness_trait'] = float(sum(i < participant_score['openness_trait'] for i in self.openness_score_list))/float(len(self.openness_score_list)-1)
            participant_percentile['conscientiousness_trait'] = float(sum(
                i < participant_score['conscientiousness_trait'] for i in self.conscientiousness_score_list))/float(len(self.conscientiousness_score_list)-1)
            participant_percentile['extraversion_trait'] = float(sum(
                i < participant_score['extraversion_trait'] for i in self.extraversion_score_list))/float(len(self.extraversion_score_list)-1)
            participant_percentile['agreeableness_trait'] = float(sum(
                i < participant_score['agreeableness_trait'] for i in self.agreeableness_score_list))/float(len(self.agreeableness_score_list)-1)
            participant_percentile['neuroticism_trait'] = float(sum(
                i < participant_score['neuroticism_trait'] for i in self.neuroticism_score_list))/float(len(self.neuroticism_score_list)-1)

            for key, val in participant_percentile.iteritems():
                if val > 0.9:
                    participant_percentile[key] = 0.9

                if val < 0.1:
                    participant_percentile[key] = 0.1

                participant_percentile[key] = int(participant_percentile[key]*100)

                if participant_percentile[key] > 75:
                    if key != 'neuroticism_trait':
                        participant_percentile[key] = '<font color="green"> Percentile ' + str(participant_percentile[key]) + '!</font>'
                    else:
                        participant_percentile[key] = '<font color="red"> Percentile ' + str(
                            participant_percentile[key]) + '!</font>'

                elif participant_percentile[key] < 25:
                    if key != 'neuroticism_trait':
                        participant_percentile[key] = '<font color="red"> Percentile ' + str(participant_percentile[key]) + '!</font>'
                    else:
                        participant_percentile[key] = '<font color="green"> Percentile ' + str(
                            participant_percentile[key]) + '!</font>'

                else:
                    participant_percentile[key] = '<font color="blue">  Percentile ' + str(
                        participant_percentile[key]) + '</font>'
            print(participant_percentile)
        return participant_score, participant_percentile

    def cal_participant_traits(self, row, cur_trait_list, ratio):
        trait_val = 0
        for cur_col in cur_trait_list:
            start_str_cur = str(cur_col) + '.'
            filter_col = [col for col in self.df if col.startswith(start_str_cur)][0]   # find col name
            trait_val += row[filter_col]
        trait_val = float(trait_val)/float(len(cur_trait_list))     # mean of traits
        trait_val = trait_val*float(ratio)                          # multiple ratio (prior)
        return trait_val

    # this provide calculating percentile later
    def calculate_all_scores(self):

        for (idx, row_participant) in self.df.iterrows():
            print(row_participant['Username'])
            cur_user_traits, blank_dict = self.calculate_individual_score(row_participant)
            self.openness_score_list.append(cur_user_traits['openness_trait'])
            self.conscientiousness_score_list.append(cur_user_traits['conscientiousness_trait'])
            self.extraversion_score_list.append(cur_user_traits['extraversion_trait'])
            self.agreeableness_score_list.append(cur_user_traits['agreeableness_trait'])
            self.neuroticism_score_list.append(cur_user_traits['neuroticism_trait'])

        self.openness_score_list.sort()
        self.conscientiousness_score_list.sort()
        self.extraversion_score_list.sort()
        self.agreeableness_score_list.sort()
        self.neuroticism_score_list.sort()
        logging.info('openness_scores' + str(self.openness_score_list))
        logging.info('conscientiousness_scores' + str(self.conscientiousness_score_list))
        logging.info('extraversion_scores' + str(self.extraversion_score_list))
        logging.info('agreeableness_scores' + str(self.agreeableness_score_list))
        logging.info('neuroticism_scores' + str(self.neuroticism_score_list))

        return

    # for each participant, send mail with his summary results
    def send_individual_mail(self):

        raise('block send mail')

        for (idx, row_participant) in self.df.iterrows():
            logging.info('Participant: ' + str(row_participant['Username']))
            if idx > 0:
               raise

            # if row_participant['Username'] not in ['guy.elad88@gmail.com', 'hharush@ebay.com']:
            #     continue
            # 'guy.elad88@gmail.com'
            # do not send mail twice
            # eBay
            if row_participant['Username'] not in ['guy.elad88@gmail.com']:
                continue

            if row_participant['Username'] in ['Hilyabr@gmail.com']:
                continue

            if row_participant['Username'] in ['hagarosental@gmail.com', 'ravidrotem@gmail.com', 'Micelad@gmail.com',
                       'Ibgd@bezeqint.net', 'kradinsky@ebay.com', 'guyel1988@gmail.com',
                       'hharush@ebay.com', 'okeynan@ebay.com', 'alnus@ebay.com', 'mlayfer@ebay.com',
                       'nkarlinski@ebay.com', 'ehacker@ebay.com', 'tsoleman@ebay.com', 'tlaor@ebay.com',
                       'seventov@ebay.com', 'tmalkai@ebay.com', 'efrayerman@ebay.com', 'rgoldberg@ebay.com',
                       'egur@ebay.com', 'rsitman@ebay.com', 'ymarkus@ebay.com', 'mszuchman@ebay.com',
                       'ikraisler@ebay.com', 'rshafir@ebay.com', 'eschichmanter@ebay.com', 'iris_mundigl@yahoo.co.uk',
                       'daniellemenuhin@gmail.com', 'ogreen@ebay.com', 'Birandan310@gmail.com', 'caharonson@ebay.com',
                       'tlazan@ebay.com', 'olibchik@ebay.com', 'dyerushalmi@ebay.com', 'rbryl@rbay.com',
                       'mebin@ebay.com', 'oyardeni@ebay.com', 'rcohenzedek@ebay.com', 'abetzaleli@ebay.com',
                       'sicohen@ebay.com', 'zmasad@ebay.com', 'lgreenblat@ebay.com', 'rtoueg@ebay.com', 'nopkanastia@gmailcom',
                       'dkesten@ebay.com', 'gmikles@ebay.com', 'bfridschtein@ebay.com', 'mmarkus@ebay.com',
                       'esorochkin@ebay.com', 'nleadner@ebay.com', 'asonnenberg@ebay.com', 'hmalul@ebay.com',
                       'nasolomon@ebay.com', 'rgilboa@ebay.com', 'abentsedef@ebay.com', 'jomendes@ebay.com',
                       'ematusov@ebay.com', 'annulechka@gmail.com', 'dbelahcen@ebay.com', 'ybarkaisygon@ebay.com',
                                               'yuval.matalon@ebay.com', 'ushtand@ebay.com', 'nshimoni@ebay.com',
                                               'rsaadon@ebay.com', 'dsahar@ebay.com']:
                continue

            # ['guy.elad88@gmail.com', 'yotamitai@gmail.com', 'ortalsen@gmail.com', 'shpshi@gmail.com', 'nir.lotan@gmail.com', 'Lior.krup@gmail.com', 'Oferst13@gmail.com', 'Oferst13@gmail.com', 'dor.ringel@gmail.com', 'kissmy@hairyass.com', 'None@none.com', 'Gadivp@walla.com', 'maoruliel@gmail.com', 'Avitalnevo@gmail.com', 'Dorzohar@gmail.com', 'Ronychanoch@gmail.com', 'Noyuzi@gmail.com', 'Guy.shoenfeld@gmail.com', 'dananuriel23@gmail.com', 'itaigol90@gmail.com', 'vipersss@walla.co.il', 'hadar.ringel@gmail.com', 't.flysher1@gmail.com', 'barak.steinmetz@gmail.com', 'oferavioz1@gmail.com', 'Roi.rabinian@gmail.com', 'Idanco40@gmail.com', 'JohnJohns@gmail.com', 'yotamzrl@gmail.com', 'jonasbrami@gmail.com', 'eladberla@gmail.com', 'mor139@hotmail.com', 'Litalfelzn@gmail.com', 'Nourtech17@gmail.com', 'xxxoooxo5@gmail.com', 'pu$$y$layer420@gmail.com', 'dvir.dukhan@campus.technion.ac.il', 'Tanya9kin@gmail.com', 'tal6el@gmail.com', 'irrelevant@gmail.com', 'howbfjla@sharklasers.com', '794orik@gmail.com', 'eliashayek1995@live.com']
            # Technion

            if row_participant['Username'] in ['yotamitai@gmail.com', 'ortalsen@gmail.com', 'shpshi@gmail.com',
                       'nir.lotan@gmail.com', 'Lior.krup@gmail.com', 'Oferst13@gmail.com', 'Oferst13@gmail.com',
                       'dor.ringel@gmail.com', 'kissmy@hairyass.com', 'None@none.com', 'Gadivp@walla.com',
                       'maoruliel@gmail.com', 'Avitalnevo@gmail.com', 'Dorzohar@gmail.com', 'Ronychanoch@gmail.com',
                       'Noyuzi@gmail.com', 'Guy.shoenfeld@gmail.com', 'dananuriel23@gmail.com', 'itaigol90@gmail.com',
                       'vipersss@walla.co.il', 'hadar.ringel@gmail.com', 't.flysher1@gmail.com',
                       'barak.steinmetz@gmail.com', 'oferavioz1@gmail.com', 'Roi.rabinian@gmail.com',
                       'Idanco40@gmail.com', 'JohnJohns@gmail.com', 'yotamzrl@gmail.com', 'jonasbrami@gmail.com',
                       'eladberla@gmail.com', 'mor139@hotmail.com', 'Litalfelzn@gmail.com', 'Nourtech17@gmail.com',
                       'xxxoooxo5@gmail.com', 'pu$$y$layer420@gmail.com', 'dvir.dukhan@campus.technion.ac.il',
                       'Tanya9kin@gmail.com', 'tal6el@gmail.com', 'irrelevant@gmail.com', 'howbfjla@sharklasers.com',
                       '794orik@gmail.com', 'eliashayek1995@live.com', 'talil@edu.haifa.ac.il', 'u12475@mvrht.net',
                       'bercovitz5@gmail.com', 'nopkanastia@gmail.com', 'nivshahar1@gmail.com', 'Shimonsheiba@gmail.com',
                       'eranaha@gmail.com', 'augu144@gmail.com', 'noaw177@gmail.com', 'Boris12345@gmail.com',
                       'orrmazor@gmail.com', 'Tomer.golany@gmail.com', 'hadarydar@gmail.com', 'Liel.mayost@gmail.com',
                       'oritorbo@gmail.com', 'Omer.amit@gmail.com', 'Gufi97@gmail.com', 'samyon.v@gmail.com',
                       'orgoldreich@gmail.com', 'klarag565@gmail.com', 'sapir.reg@campus.technion.ac.il',
                       'Hamody315@gmail.com', 'Saharsela271@gmail.com', 'Avishaya67@gmail.com', 'Saifunny@gmail.com',
                       '123@123.com', 'Michal.inbar0@gmail.com', 'Hilyabr@gmail.com', 'izikasch@hotmail.com',
                                               'sylilit@campus.technion.ac.il', 'Hilyabr@gmail.com',
                                               'sivanr86@gmail.com', 'Shiharel1@gmail.com',
                                               'shai98m@gmail.com', 'Sdg9595@hotmail.com', 'atorbiner@gmail.com', 'galbh33@gmail.com']:
                continue

            # get participant score
            cur_user_traits, participant_percentile = self.calculate_individual_score(row_participant, True)      # return participant scores

            # create plot
            plot_file_name = self.create_plot_comparison_traits(cur_user_traits, row_participant)

            # create a mail
            from email.MIMEMultipart import MIMEMultipart
            from email.MIMEText import MIMEText
            from email.MIMEImage import MIMEImage

            # Define these once; use them twice!

            strFrom = 'guy.elad88@gmail.com'
            strTo = row_participant['Username']     # participant mail
            # strTo = 'tomer.golany@gmail.com'

            # Create the root message and fill in the from, to, and subject headers
            msgRoot = MIMEMultipart('related')
            msgRoot['Subject'] = 'Personality Test - Results'
            msgRoot['From'] = strFrom
            msgRoot['To'] = strTo
            msgRoot.preamble = 'This is a multi-part message in MIME format.'

            # Encapsulate the plain and HTML versions of the message body in an
            # 'alternative' part, so message agents can decide which they want to display.
            msgAlternative = MIMEMultipart('alternative')
            msgRoot.attach(msgAlternative)

            msgText = MIMEText('This is the alternative plain text message.')
            msgAlternative.attach(msgText)

            html = """\
                        <html dir="ltr" lang="en">
                          <head>
                          </head>
                          <body dir="ltr">
                          <p>
                            <h3>Dear participant, Thank you for completing our survey! </h3> <br> 
                                Following the answers we received from you, please see your personality analysis below, (please also see images attached) <br>
                                <ol>
                                <li> <b> Openness: </b> """ + str(round(cur_user_traits['openness_trait'], 2)) + """ - """ + str(participant_percentile['openness_trait']) + """ </li><br> 
                                <li><b> Conscientiousness: </b> """ + str(round(cur_user_traits['conscientiousness_trait'], 2)) + """ - """ + str(participant_percentile['conscientiousness_trait']) + """ </li><br>
                                <li> <b> Extraversion: </b> """ + str(round(cur_user_traits['extraversion_trait'], 2)) + """ - """ + str(participant_percentile['extraversion_trait']) + """ </li><br> 
                                <li><b> Agreeableness: </b> """ + str(round(cur_user_traits['agreeableness_trait'], 2)) + """ - """ + str(participant_percentile['agreeableness_trait']) + """ </li><br>
                                <li> <b> Neuroticism: </b> """ + str(round(cur_user_traits['neuroticism_trait'], 2)) + """ - """ + str(participant_percentile['neuroticism_trait']) + """ </li><br> 
                                </ol> 

                                Here is a <a href="https://en.wikipedia.org/wiki/Big_Five_personality_traits">link</a> 
                                with more data about the Big Five Personality traits.
                            
                            </p> <br>
                            
                            <img src="cid:image1", height="300" width="450">
                            <img src="cid:image2", height="300" width="456">
                          </body>
                        </html>
                        """

            # We reference the image in the IMG SRC attribute by the ID we give it below
            # msgText = MIMEText('<b>Yor are my lion <i></i></b> <br><img src="cid:image1"><br>Nifty!', 'html')
            msgText = MIMEText(html, 'html')
            msgAlternative.attach(msgText)


            # NEW
            # This example assumes the image is in the current directory
            fp = open(plot_file_name, 'rb')
            msgImage = MIMEImage(fp.read())
            fp.close()

            # Define the image's ID as referenced above
            msgImage.add_header('Content-ID', '<image1>')
            msgRoot.attach(msgImage)

            fp = open('/Users/sguyelad/PycharmProjects/research/survey_pilot/participant_plot/trait table meaning.jpg',
                      'rb')
            msgImageExplain = MIMEImage(fp.read())
            fp.close()

            msgImageExplain.add_header('Content-ID', '<image2>')
            msgRoot.attach(msgImageExplain)

            '''
            # OLD
            fp = open(plot_file_name, 'rb')
            msgImagePlot = MIMEImage(fp.read())
            fp.close()

            # Define the image's ID as referenced above
            msgImagePlot.add_header('Personality plot results', '<{}>'.format(plot_file_name))
            msgRoot.attach(msgImagePlot)

            fp = open('/Users/sguyelad/PycharmProjects/research/survey_pilot/participant_plot/trait table meaning.jpg', 'rb')
            msgImageExplain = MIMEImage(fp.read())
            fp.close()

            # Define the image's ID as referenced above
            msgImageExplain.add_header('Traits meaning', '<image1>')
            msgRoot.attach(msgImageExplain)
            '''

            html = """
            <html lang="en">
            <head>
            <style>
            img { 
                width:100%; 
            }
            </style>
              <title>Bootstrap Example</title>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
              <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
              <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
            </head>
            <body>
            <div>
            <img src='/Users/sguyelad/PycharmProjects/research/survey_pilot/participant_plot/trait table meaning.jpg' alt="traits table">
            </div>
            </body>
            </html>
            """



            # Send the email (this example assumes SMTP authentication is required)
            import smtplib

            fromaddr = 'guy.elad88@gmail.com'
            toaddrs = strTo
            username = 'guy.elad88@gmail.com'
            password = 200600716

            server = smtplib.SMTP('smtp.gmail.com:587')
            server.ehlo()
            server.starttls()
            server.login(username, password)
            server.sendmail(fromaddr, toaddrs, msgRoot.as_string())
            server.quit()

            logging.info('send mail to : ' + str(toaddrs))
            print('send mail to : ' + str(toaddrs))

        return

    # create plot compare participant traits and average traits
    def create_plot_comparison_traits(self, cur_user_traits, row_participant):

        # data to plot
        n_groups = 5
        cur_participant_score = (cur_user_traits['openness_trait'],
                                 cur_user_traits['conscientiousness_trait'],
                                 cur_user_traits['extraversion_trait'],
                                 cur_user_traits['agreeableness_trait'],
                                 cur_user_traits['neuroticism_trait'])

        max_person_trait = cur_user_traits[max(cur_user_traits.keys(), key=(lambda k: cur_user_traits[k]))]
        min_person_trait = cur_user_traits[min(cur_user_traits.keys(), key=(lambda k: cur_user_traits[k]))]

        max_ebay_traits = max(self.ebay_avg_openness,
                       self.ebay_avg_conscientiousness,
                       self.ebay_avg_extraversion,
                       self.ebay_avg_agreeableness,
                       self.ebay_avg_neuroticism)

        min_ebay_traits = min(self.ebay_avg_openness,
                              self.ebay_avg_conscientiousness,
                              self.ebay_avg_extraversion,
                              self.ebay_avg_agreeableness,
                              self.ebay_avg_neuroticism)

        max_in_plot = max(max_person_trait, max_ebay_traits)
        min_in_plot = min(min_person_trait, min_ebay_traits)

        means_score = (self.ebay_avg_openness,
                       self.ebay_avg_conscientiousness,
                       self.ebay_avg_extraversion,
                       self.ebay_avg_agreeableness,
                       self.ebay_avg_neuroticism)

        # create plot
        fig, ax = plt.subplots(figsize=(9, 6))
        index = np.arange(n_groups)
        bar_width = 0.30
        opacity = 0.65

        rects1 = plt.bar(index, cur_participant_score, bar_width,
                         alpha=opacity,
                         color='b',
                         label='Your score')

        rects2 = plt.bar(index + bar_width, means_score, bar_width,
                         alpha=opacity,
                         color='g',
                         label='Average score')
                         # label='Average eBay employees score')

        plt.xlabel('Personality traits')
        plt.ylabel('Scores')
        plt.title('Personality Test Results - ' + str(row_participant['Full Name']))
        plt.xticks(index + bar_width,
                   ('Openness to experience', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'),
                   rotation=20)

        axes = plt.gca()

        axes.set_ylim([max(0, min_in_plot-10), min(100, max_in_plot+10)])

        # plt.legend()

        # ax.legend((rects1[0], rects2[0]), ('Your score', 'Average eBay employees score'))
        ax.legend((rects1[0], rects2[0]), ('Your score', 'Average score'))

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        # plt.show()

        plot_name = '/Users/sguyelad/PycharmProjects/research/survey_pilot/participant_plot/' + \
                    str(row_participant['Full Name']) + '.png'
        plt.savefig(plot_name, bbox_inches='tight')

        plt.close()
        return plot_name


def main(name):
    calculate_obj = CalculateScore()            # create object and variables
    calculate_obj.init_debug_log()              # init log file
    calculate_obj.load_clean_csv_results()      # load dataset
    calculate_obj.change_reverse_value()        # change specific column into reverse mode
    calculate_obj.calculate_average_score()     # calculate average score for the big five traits
    calculate_obj.calculate_all_scores()        #

    # to send an email response with results
    # calculate_obj.send_individual_mail()

if __name__ == '__main__':
    main('guy')




