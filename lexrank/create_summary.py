

# class Summarization produce summary from personality-based-LexRank results
# main arguments contain:
#       multi_document_summarization = 'single'         # summarization method for single/multi documents
#       summarization_similarity_threshold              # between zero to one
#       summarization_version = 'top_relevant'          # 'top_relevant', 'Bollegata', 'Shahaf'
class Summarization:

    def __init__(self, logging, multi_document_summarization, summarization_similarity_threshold,
                 summarization_version, similarity_matrix_unnormalized, sorted_ix, summary_size, sentences):

        self.logging = logging
        self.multi_document_summarization = multi_document_summarization
        self.summarization_similarity_threshold = summarization_similarity_threshold
        self.summarization_version = summarization_version
        self.similarity_matrix_unnormalized = similarity_matrix_unnormalized
        self.sorted_ix = sorted_ix          # indexes sorted
        self.summary_size = summary_size
        self.sentences = sentences          # sentences not sorted

        # class objects
        self.sentence_already_inserted = None
        self.discarded_sentences = None
        self.description_summary_list = None        # object used to retrieve the final summarization

    # create summary (main function)
    def create_summary(self):

        description_summary_list = list()

        if self.multi_document_summarization == 'single':
            description_summary_list = self.single_document_summarization()

        elif self.multi_document_summarization == 'multi':
            if self.summarization_version == 'top_relevant':
                description_summary_list = self.top_relevant_summarization()
            elif self.summarization_version == 'Bollegata':
                description_summary_list = self.bollegata_four_expert_summarization()
                raise ('TODO add Bollegata version')
            elif self.summarization_version == 'Shahaf':
                raise ('TODO add Shahaf version')

        self.description_summary_list = description_summary_list

        return

    # single document summarization (summary by sentence order in description with threshold condition)
    def single_document_summarization(self):

        self.logging.info('')
        self.logging.info('start single document summarization')
        self.logging.info('summarization similarity threshold: ' + str(self.summarization_similarity_threshold))
        self.logging.info('max similarity between sentences: ' + str(round(self.find_max_similarity(), 3)))

        # 1. check by importance order (stationary distribution) which sentences are too similar
        # 2. keep only numbers of final sentences
        # 3. output sentences by original order in single description

        description_summary_list = list()
        self.sentence_already_inserted = list()

        # sentences which discarded from summarization due to similarity above threshold
        self.discarded_sentences = list()

        for sentence_idx in self.sorted_ix:

            # check if we have already enough sentences
            if len(self.sentence_already_inserted) >= self.summary_size:
                break

            # 1. check if similarity with previous sentences is below threshold
            cur_max_similarity = self.check_cur_max_similarity(sentence_idx)

            # max similarity above threshold
            if cur_max_similarity > self.summarization_similarity_threshold:
                self.discarded_sentences.append(sentence_idx)
                self.logging.info(
                    'Sentence ' + str(sentence_idx) + ' above threshold - therefore discarded from summary - ' + str(
                        round(cur_max_similarity, 2)))
                continue

            # 2. max similarity below threshold
            else:
                self.sentence_already_inserted.append(sentence_idx)

        # 3. output sentences by original order in single description
        # first we sort sentences
        self.sentence_already_inserted.sort()

        # output sentences into final description by original order in the singe order
        for sentence_idx in self.sentence_already_inserted:
            description_summary_list.append(self.sentences[sentence_idx])

        return description_summary_list

    # extract summarization using top-relevant
    # like single summarization expect sort sentences due to their original order
    def top_relevant_summarization(self):

        self.logging.info('')
        self.logging.info('summarization version: top-relevant')
        self.logging.info('summarization similarity threshold: ' + str(self.summarization_similarity_threshold))

        self.logging.info('max similarity between sentences: ' + str(round(self.find_max_similarity(), 3)))

        description_summary_list = list()
        self.sentence_already_inserted = list()

        # sentences which discarded from summarization due to similarity above threshold
        self.discarded_sentences = list()

        for sentence_idx in self.sorted_ix:

            # check if we have already enough sentences
            if len(self.sentence_already_inserted) >= self.summary_size:
                break

            # check if similarity with previous sentences is below threshold
            cur_max_similarity = self.check_cur_max_similarity(sentence_idx)

            # max similarity above threshold
            if cur_max_similarity > self.summarization_similarity_threshold:
                self.discarded_sentences.append(sentence_idx)
                self.logging.info('Sentence ' + str(sentence_idx) + ' above threshold - therefore discarded from summary - ' + str(round(cur_max_similarity, 2)))
                continue

            # max similarity below threshold
            else:
                description_summary_list.append(self.sentences[sentence_idx])
                self.sentence_already_inserted.append(sentence_idx)

        return description_summary_list

    # summarization method for multi-documents using 4 experts version and greedy algorithm
    def bollegata_four_expert_summarization(self):
        raise('currently we do not support Bollegata algorithm')
        description_summary_list = list()
        return description_summary_list

    # find max similarity between two sentences
    def find_max_similarity(self):
        import numpy as np
        similarity_list = np.asarray(self.similarity_matrix_unnormalized).reshape(-1)
        similarity_list = filter(lambda a: a != 1, similarity_list)
        return max(similarity_list)

    # check max similarity with respect to current sentence
    def check_cur_max_similarity(self, sentence_idx):
        cur_max_similarity = 0.0
        similarity_row_list = self.similarity_matrix_unnormalized[sentence_idx]

        # check max similarity with respect to sentence already inserted into summary
        for inserted_sen_idx in self.sentence_already_inserted:
            if similarity_row_list[inserted_sen_idx] > cur_max_similarity:
                cur_max_similarity = similarity_row_list[inserted_sen_idx]

        return cur_max_similarity







