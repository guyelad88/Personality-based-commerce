
# indicate which pre-processing steps needs to do
pre_processing_configs = {

    'pre_process_description': True,    # clean description (e.g. w_1.w_2 w_1w_2)
    'filter_description': True,
    'remove_row_duplication': True,
    'balance_description_per_user': True,

    'POS_filter': True,
    'Number_filter': False,             # TODO add this

    'analyze_PT_groups': True,          # add 'L'/'M'/'H' to each PT

    'calculate_KL': True
}

# tune duplication validity and user name validity
extract_big_five_inventory_score = {
    'threshold': 0.15,                       # max diff in trait withing user with duplication
    'name_length_threshold': 7              # min name length
}

# determine how to filter descriptions
filter_description = {
    'MAX_LENGTH': 1000,
    'MIN_LENGTH': 5,
    'DROP_NA': True,
    'DROP_MIN': True,
    'DROP_MAX': True,
    'DROP_NON_ENGLISH': True,           # remain only desc in english
    'DROP_NON_ENGLISH_WORDS': True,     # remain only valid words in english
    'DROP_DUPLICATION': True,
    'FLAG_UNINFORMATIVE_WORDS': True,

    'UNINFORMATIVE_WORDS': ['ship', 'accept', 'positive', 'contact', 'payment', 'address', 'received', 'reply',
                            'shipping', 'sign', 'addresses', 'purchase', 'fees', 'please', 'bid', 'days', 'bidding',
                            'cost', 'buyer', 'delivery', 'shipping', 'payment', 'address', 'days', 'included', 'zone',
                            'response', 'time', 'country', 'arrival', 'returned', 'manufacturer', 'follow', 'shipped',
                            'express', 'transaction', 'div', 'text', 'die', 'hat', 'sold', 'return', 'shipment', 'via',
                            'return', 'warranty', 'defect', 'order', 'repair', 'agree', 'ist', 'es', 'sie', 'wir',
                            'hen', 'sind', 'den', 'ber', 'dye', 'pal', 'nach', 'pal', 'es', 'ber', 'das', 'tie',
                            'policies', 'package', 'amp', 'http', 'www', 'gif', 'customer', 'feedback', 'seller',
                            'understanding', 'refund', 'paying', 'receive', 'delay', 'post', 'following',
                            'information', 'mail', 'telephone', 'de', 'en', 'la', 'thanks', 'sellers'],

    'BI_GRAM_UNINFORMATIVE_WORDS': ['our store', 'thank you', 'we stand', 'solve your problem', 'business day',
                                    'we shall', 'solve your problem', 'may little different',
                                    'different on your computer', 'monitor setting', 'computer monitor',
                                    'welcome to my store', 'mobile phone', 'our price', 'our warehouse',
                                    'we do not work', 'we will', 'we are', 'terms of sale', 'we understand',
                                    'place your orders', 'we can assure', 'these discounts', 'item purchases',
                                    '30 days', 'money back'],

    'VERTICAL': None    # 'Fashion'   # None                    # None/Fashion/Electronics e.g
}

# POS properties
POS = {
    'VALID_POS': ['JJ', 'JJR', 'JJS'],  # ['RBS', 'RB', 'RBR', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNP', 'NNPS'],
    'filter_pos_flag': True,    # save in addition only a specific list of POS
    'save_pos': True            # save POS or words
}

# indicate how to build the vocabularies (e.g. extroversion vs. introversion)
create_vocabularies = {
    'gap_value': 0.4,  # must be a float number between zero to one
    # 'vocabulary_method': 'documents',       # 'documents', 'aggregation'
    # 'split_method': 'traits',               # 'vertical', 'traits', 'traits_vertical'
    # 'vertical': '',                         # 'Fashion'
}

calculate_kl = {
    'TOP_K_WORDS': 50,           # present top words
    'SMOOTHING_FACTOR': 1.0,     # smoothing factor for calculate term contribution
    'NGRAM_RANGE': (1, 1),
    'VOCABULARY_METHOD': 'documents',    # 'documents', 'aggregation'
    'CONTRIBUTE_TYPE': 'description_fraction',      # 'buyers_fraction'
    'CLIP_DESCRIPTION': 5,  # max desc contain token per user
    'NORMALIZE_CONTRIBUTE': {
        'flag': False,
        'type': 'min_max',      # ratio'
    },
    'FIND_WORD_DESCRIPTION': {  # save top words with k description they appear in
        'flag': True,
        'k': 20
    },
    'KL_TYPE': {
        'type': 'words',                    # 'words', 'POS'
        'column_name': 'description'     # 'description_POS_str', 'description', 'description_filter_words_str'
    }
}

personality_trait = ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']

balance_description = {
    'percentile_truncated': 95,
    'max_descriptions': 35                          # None
}

test_lexrank = {
    'summary_size': 'max',                          # summarization length - 'max' or int
    'HTML_summary_size': 3,
    'threshold': 0.03,                              # min edge weight between two sentences - below remove the edge

    # for vanilla-LexRank set damping factor to 0.1-0.2
    # for personality-based-LexRank set damping factor to 0.01-0.3 (SM: probability not to jump)
    'damping_factor': 0.1,

    'summarization_similarity_threshold': 0.55,      # remove from summarization if the similarity is above
    'personality_trait_dict': {
        'agreeableness': 'H',
        'conscientiousness': 'H',
        # 'extraversion': 'H',
        # 'openness': 'H',
    },
    'lex_rank_algorithm_version': 'personality-based-LexRank',      # 'vanilla-LexRank', 'personality-based-LexRank'
    'products_ids': ['B0746GQ56P'],     # id 16: 'B00PQ5UH0C'],      # products id to create summary for them

    'corpus_size': 200,            # 'max'/1000 - influence IDF - high number is leading to high computation time

    'target_sentences_length': {
        'min': 0,
        'max': 100
    },

    # please don't change (damping factor:1 same effect)
    'personality_word_flag': True,
    'random_walk_flag': True,                        # flag if combine random jump between sentences

    'multi_document_summarization': 'single',                       # summarization method for single/multi documents

    'summarization_version': 'top_relevant',                        # 'top_relevant', 'Bollegata', 'Shahaf'


    # current user personality - 'H'\'L'\'M' (high\low\miss(or mean)
    # TODO change it isn't the description we need to use
    'corpus_path_file': '../data/descriptions_data/1425 users input/merge_20048.csv',  # calculate idf from

    # 'target_item_description_file': '../data/amazon_description/size=435_min_size=10_time=2018-08-23 12:41:02.csv',  # amazon_949

    # clean descriptions
    # 'target_item_description_file': '../data/amazon_description/experiment_input_clean_description.csv',
    'target_item_description_file': '../data/amazon_description/experiment_input_clean_description.xlsx',

    'trait_relative_path_dict': '../results/data/kl/uni-gram regular 2018-08-21 21:04:59/all_words_contribute',

}