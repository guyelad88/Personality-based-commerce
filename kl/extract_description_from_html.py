import sys
import csv
import os
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

        # lod_file_name = self.log_dir + 'predict_personality_from_desc_' + str(self.cur_time) + '.log'

        log_file_name = self.log_dir + '_extract_description_from_html_' + str(self.cur_time) + '.log'

        # logging.getLogger().addHandler(logging.StreamHandler())

        logging.basicConfig(filename=log_file_name,
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
        html_output_dir = "/Users/gelad/Desktop/html_example/{}/".format(self.cur_time)
        if not os.path.exists(html_output_dir):
            os.makedirs(html_output_dir)

        from bs4 import BeautifulSoup
        count_id = 0
        for item_id, description_string in self.item_description_dict.iteritems():

            if item_id in ['332371767234', '152361715916']:
                continue

            # if item_id not in ['232690299503']:  # '292198425550']:
            #     continue

            count_id += 1

            logging.info("Count: " + str(count_id) + ", Item id: " + str(item_id))
            # soup = BeautifulSoup(description_string)

            soup = BeautifulSoup(description_string, 'html.parser')


            """
            print(soup.prettify())
            print('')
            print('')
            print('')
            print('')
            print('')
            """

            # print('finish pretty')

            # kill all script and style elements
            # for script in soup(["script", "style"]):
            #     script.extract()  # rip it out

            result = list()
            for tag in soup.findAll(True, {'id': True}):
                result.append(tag['id'])

            print('all id: {}'.format(result))
            result = [x for x in result if 'desc' in x]
            print('all id: {}'.format(result))

            obj = list()
            for id_name in result:
                if soup.find_all(id=id_name) is not None:
                    obj.extend(soup.find_all(id=id_name))

            text_list = list()

            for o in obj:
                new_text = o.get_text()

                for text in text_list:
                    if new_text in text:
                        continue
                text_list.append(new_text)

            final_text = '. '.join(text_list)
            print('')
            print('new method')
            print(final_text)

            soup = BeautifulSoup(description_string)
            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out
            text = soup.get_text()

            print('')
            print('old method')
            print(text)

            if len(text) > 300:
                print('item: {}, length {}'.format(item_id, len(text)))
                soup = BeautifulSoup(description_string, 'html.parser')
                print(soup.prettify())
                beauty = u''.join(soup.prettify()).encode('utf-8').strip()
                html_str = '<html>{}</html>'.format(beauty)

                file_full_path = "{}{}_{}_{}.html".format(html_output_dir, len(text), count_id, item_id)
                Html_file = open(file_full_path, "w")

                print('save full path: {}'.format(file_full_path))

                Html_file.write(html_str)
                Html_file.close()
            # get text
            # text = soup.get_text()
            if count_id > 1000:
                raise()

            continue
            # remove more HTML which remain after soup cleaning
            from HTMLParser import HTMLParser

            class MLStripper(HTMLParser):
                def __init__(self):
                    self.reset()
                    self.fed = []

                def handle_data(self, d):
                    self.fed.append(d)

                def get_data(self):
                    return ''.join(self.fed)

            s = MLStripper()
            s.feed(text)
            text = s.get_data()

            # remove non-ascii characters
            text = ''.join([i if ord(i) < 128 else ' ' for i in text])

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            text.encode('utf-8')
            text.replace('\n', ' ')

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
            if not os.path.exists(self.directory_output):
                os.makedirs(self.directory_output)

            file_name = self.directory_output + 'num_items_' + str(len(self.item_text_dict)) + '_' + str(self.cur_time) + '.csv'
            with open(file_name, 'wb') as f:
                w = csv.writer(f)
                w.writerow(['item_id', 'description'])
                w.writerows(self.item_text_dict.items())

            logging.info('File saved: ' + file_name)

        return


def main(description_file, log_dir, directory_output, verbose_flag):

    extract_descriptions_obj = ExtractDescriptions(description_file, log_dir, directory_output, verbose_flag)

    extract_descriptions_obj.init_debug_log()                      # init log file
    extract_descriptions_obj.load_clean_csv_results()              # load data set
    extract_descriptions_obj.extract_words()
    extract_descriptions_obj.print_save_text()


if __name__ == '__main__':

    # input file name
    description_file = '/Users/gelad/Personality-based-commerce/data/descriptions_data/desc.tsv'        # part 1
    description_file = '/Users/gelad/Personality-based-commerce/data/descriptions_data/desc1.tsv'     # part 2

    log_dir = 'log/'
    directory_output = './results/descriptions_clean/'
    verbose_flag = True
    main(description_file, log_dir, directory_output, verbose_flag)
