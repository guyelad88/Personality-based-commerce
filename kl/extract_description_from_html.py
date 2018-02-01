import sys
import csv
import logging


# target: extract from csv/tsv file all description from HTML code
# output: csv file contain item_id and his description
class ExtractDescriptions:

    def __init__(self, description_file, log_dir, directory_output, verbose_flag):

        self.description_file = description_file    # description file
        self.log_dir = log_dir                      # log directory
        self.directory_output = directory_output    # save description texts
        self.verbose_flag = verbose_flag            # print results in addition to log file

        self.item_description_dict = dict()         # dictionary contain id and html code
        self.item_text_dict = dict()                # dictionary contain id and text extracted from html code

        self.cur_time = str
        csv.field_size_limit(sys.maxsize)

    # build log object
    def init_debug_log(self):
        import logging
        import time
        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        lod_file_name = self.log_dir + 'analyze_description_' + str(self.cur_time) + '.log'

        # logging.getLogger().addHandler(logging.StreamHandler())

        logging.basicConfig(filename=lod_file_name,
                            # filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        # print result in addition to log file
        if self.verbose_flag:
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

    def load_clean_csv_results(self):

        description_counter = 0     # counter
        cur_key = str               # cur item id

        with open(self.description_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')

            for row in reader:

                if len(row) > 1 and '((((((((((' in row[1]:
                    description_counter += 1
                    cur_key = row[0]
                    self.item_description_dict[cur_key] = ''
                    self.item_description_dict[cur_key] += str(row[1])
                else:
                    if len(row) > 0:
                        for str_cell in row:
                            self.item_description_dict[cur_key] += str(str_cell)

        logging.info('Total description: ' + str(description_counter))

        # delete first and last brackets (((( ))))
        for item_id, description_string in self.item_description_dict.iteritems():
            # print(self.item_description_dict[item_id])
            self.item_description_dict[item_id] = self.item_description_dict[item_id][10:]
            self.item_description_dict[item_id] = self.item_description_dict[item_id][:-10]
            # print(self.item_description_dict[item_id])

    def extract_words(self):

        import urllib
        from bs4 import BeautifulSoup
        count_id = 0
        for item_id, description_string in self.item_description_dict.iteritems():

            if item_id in ['332371767234', '152361715916']:
                continue

            count_id += 1
            # limit number of output descriptions
            # if count_id > 10:
            #     break

            logging.info("Count: " + str(count_id) + ", Item id: " + str(item_id))
            # url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
            # html = urllib.urlopen(url).read()
            soup = BeautifulSoup(description_string)

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            text.encode('utf-8')
            text.replace('\n', ' ')
            '''text = lambda s: text.decode('utf8', 'ignore')
            text.replace('\xc2', ' ')
            text.replace('\xa0', ' ')'''
            self.item_text_dict[item_id] = text.encode('utf-8')
        return

    def print_save_text(self, print_flag=False, save_flag=True):

        # print results
        if print_flag:
            for item_id, cur_text in self.item_text_dict.iteritems():
                logging.info('Item id: ' + str(item_id))
                logging.info('Item text: ' + str(cur_text))

        # save descriptions texts into csv
        if save_flag:
            file_name = self.directory_output + 'num_items_' + str(len(self.item_text_dict)) + '_' + str(self.cur_time) + '.csv'
            logging.info('File saved: ' + file_name)
            with open(file_name, 'wb') as f:
                w = csv.writer(f)
                w.writerow(['item_id', 'description'])
                w.writerows(self.item_text_dict.items())

        return


def main(description_file, log_dir, directory_output, verbose_flag):

    extract_descriptions_obj = ExtractDescriptions(description_file, log_dir, directory_output, verbose_flag)

    extract_descriptions_obj.init_debug_log()                      # init log file
    extract_descriptions_obj.load_clean_csv_results()              # load data set
    extract_descriptions_obj.extract_words()
    extract_descriptions_obj.print_save_text()

if __name__ == '__main__':

    # input file name
    description_file = '/Users/sguyelad/PycharmProjects/research/data/descriptions_data/descriptions.tsv'
    description_file = '/Users/sguyelad/PycharmProjects/research/data/descriptions_data/2017_desc.csv'
    log_dir = '/Users/sguyelad/PycharmProjects/research/kl/log/'
    directory_output = '/Users/sguyelad/PycharmProjects/research/kl/descriptions/'
    verbose_flag = True
    main(description_file, log_dir, directory_output, verbose_flag)
