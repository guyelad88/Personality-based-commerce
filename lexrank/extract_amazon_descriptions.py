import logging
import pandas as pd


class ExtractDescriptionFromAmazon:

    def __init__(self, amazon_crawl_path, amazon_crawl_path_2, amazon_crawl_path_3):

        self.amazon_crawl_path = amazon_crawl_path
        self.amazon_crawl_path_2 = amazon_crawl_path_2
        self.amazon_crawl_path_3 = amazon_crawl_path_3
        self.file_path_list = [self.amazon_crawl_path_3, self.amazon_crawl_path_2, self.amazon_crawl_path]
        self.log_dir = 'log/'
        self.verbose_flag = True
        self.amazon_desc_df = None
        self.df_description = None

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.word_cont_dict = dict()    # word and correspond contribute value
        return

    # build log object
    def init_debug_log(self):
        import logging

        lod_file_name = self.log_dir + 'Extract_Description' + str(self.cur_time) + '.log'

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

    # extract description
    def extract_description(self):
        logging.info('')
        logging.info('start to extract description')

        # build CSV file
        self.df_description = pd.DataFrame(columns=['ID', 'URL', 'TITLE', 'DESCRIPTION_1', 'DESCRIPTION_2', 'TEXT_LENGTH'])

        for file_index, cur_file_path in enumerate(self.file_path_list):
            logging.info('load file: ' + str(cur_file_path))
            inserted_items = 0
            self.amazon_desc_df = pd.read_excel(open(cur_file_path, 'rb'), sheet_name=0)   # open excel file

            for index, cur_row in self.amazon_desc_df.iterrows():

                # check if product has text data
                if not pd.isnull(cur_row['TEXT']) or not pd.isnull(cur_row['MORE_TEXT']):

                    # build item description
                    full_text = ''
                    desc_1 = ''
                    desc_2 = ''
                    if not pd.isnull(cur_row['TEXT']):
                        full_text += cur_row['TEXT'].encode('ascii', 'ignore').decode('ascii')
                        full_text += '.'
                        desc_1 = cur_row['TEXT'].encode('ascii', 'ignore').decode('ascii')
                    if not pd.isnull(cur_row['MORE_TEXT']):
                        full_text += cur_row['MORE_TEXT'].encode('ascii', 'ignore').decode('ascii')
                        desc_2 = cur_row['TEXT'].encode('ascii', 'ignore').decode('ascii')

                    text_length = len(full_text.split('.'))

                    try:
                        # insert item into DF
                        self.df_description.loc[len(self.df_description)] = [
                            cur_row['ID'].encode('ascii', 'ignore').decode('ascii'),
                            cur_row['URL'].encode('ascii', 'ignore').decode('ascii'),
                            cur_row['TITLE'].encode('ascii', 'ignore').decode('ascii'),
                            desc_1,
                            desc_2,
                            text_length
                        ]
                        inserted_items += 1
                        logging.info('description inserted len: ' + str(text_length))
                    except:
                        pass

            logging.info('finish load file: ' + str(cur_file_path) + ', item inserted: ' + str(inserted_items))

            # save file after any additional input file
            output_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/amazon_description/' + \
                          'amazon' + '_' + str(self.df_description.shape[0]) + '.csv'
            self.df_description.to_csv(output_file, index=False)
            logging.info('save file: ' + str(output_file) + ', total items: ' + str(self.df_description.shape[0]))

        # save DF into CSV
        output_file = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/amazon_description/' + 'amazon.csv'

        self.df_description.to_csv(output_file, index=False)
        logging.info('save file: ' + str(output_file) + ', total items: ' + str(self.df_description.shape[0]))
        return


def main(amazon_crawl_data, amazon_crawl_data_2, amazon_crawl_data_3):
    ExtractDescriptionObj = ExtractDescriptionFromAmazon(amazon_crawl_data, amazon_crawl_data_2, amazon_crawl_data_3)
    ExtractDescriptionObj.init_debug_log()
    ExtractDescriptionObj.extract_description()

# TODO change to CSV files for better performance
if __name__ == '__main__':
    amazon_crawl_data = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/amazon_description/amazon-crawl-output.xlsx'
    amazon_crawl_data_2 = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/amazon_description/amazon-crawl-output-2.xlsx'
    amazon_crawl_data_3 = '/Users/sguyelad/PycharmProjects/Personality-based-commerce/data/amazon_description/amazon-crawl-output-3.xlsx'
    main(amazon_crawl_data, amazon_crawl_data_2, amazon_crawl_data_3)