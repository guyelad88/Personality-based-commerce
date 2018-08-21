
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
    'threshold': 0.8,                       # max diff in trait withing user with duplication
    'name_length_threshold': 5              # min name length
}

# determine how to filter descriptions
filter_description = {
    'MAX_LENGTH': 1000,
    'MIN_LENGTH': 10,
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
                            'information', 'mail', 'telephone'],

    'BI_GRAM_UNINFORMATIVE_WORDS': ['our store', 'thank you', 'we stand', 'solve your problem', 'business day',
                                    'we shall', 'solve your problem', 'may little different',
                                    'different on your computer', 'monitor setting', 'computer monitor',
                                    'welcome to my store', 'mobile phone', 'our price', 'our warehouse',
                                    'we do not work', 'we will', 'we are'],

    'VERTICAL': None                    # None/Fashion/Electronics e.g
}

# POS properties
POS = {
    'VALID_POS': ['JJ', 'JJR', 'JJS'], # ['RBS', 'RB', 'RBR', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNP', 'NNPS'],  # save only this POS in addition
    'filter_pos_flag': True,    # save in addition only a specific list of POS
    'save_pos': True            # save POS or words
}

# indicate how to build the vocabularies (e.g. extroversion vs. introversion)
create_vocabularies = {
    'gap_value': 0.4,  # must be a float number between zero to one
    # 'vocabulary_method': 'documents',       # 'documents', 'aggregation'
    # 'split_method': 'traits',               # 'vertical', 'traits', 'traits_vertical'
    # 'personality_trait': 'extraversion',  # 'agreeableness' 'extraversion' 'openness' 'conscientiousness' 'neuroticism'
    # 'vertical': '',                         # 'Fashion'
}

calculate_kl = {
    'TOP_K_WORDS': 50,           # present top words
    'SMOOTHING_FACTOR': 1.0,     # smoothing factor for calculate term contribution
    'NGRAM_RANGE': (1, 1),
    'VOCABULARY_METHOD': 'documents',    # 'documents', 'aggregation'
    'NORMALIZE_CONTRIBUTE': {
        'flag': False,
        'type': 'min_max',      # ratio'
    },
    'FIND_WORD_DESCRIPTION': {  # save top words with k description they appear in
        'flag': True,
        'k': 200
    },
    'KL_TYPE': {
        'type': 'words',                    # 'words', 'POS'
        'column_name': 'description'     # 'description_POS_str', 'description', 'description_filter_words_str'
    }
}

personality_trait = ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']

balance_description = {
    'percentile_truncated': 95,
    'max_descriptions': 35  # None
}

test_lexrank= {
    'summary_size': 10,             # summarization length - 'max' or int
    'threshold': 0.03,             # min edge weight between two sentences
    'damping_factor': 0.8,        # probability not to jump
    'summarization_similarity_threshold': 0.4,
    'target_sentences_length': {
        'min': 0,
        'max': 100
    },
    'corpus_size': 100,  # 'max'          # TODO be careful using this value - limit idf computation time
    # please don't change (damping factor:1 same effect)
    'personality_word_flag': True,
    'random_walk_flag': True,                        # flag if combine random jump between sentences

    'multi_document_summarization': 'single',             # summarization method for single/multi documents
    'lex_rank_algorithm_version': 'personality-based-LexRank',      # 'vanilla-LexRank', 'personality-based-LexRank'
    'summarization_version': 'top_relevant',              # 'top_relevant', 'Bollegata', 'Shahaf'

    # current user personality - 'H'\'L'\'M' (high\low\miss(or mean))
    'personality_trait_dict': {
        'openness': 'L',
        'conscientiousness': 'L',
        'extraversion': 'L',
        'agreeableness': 'L',
        'neuroticism': 'L'
    }
}


