
# indicate which pre-processing steps needs to do
pre_processing_configs = {
    'remove_duplication': True,         # currently under extract_bfi_score
    'remove_fake_users': True,          # currently under extract_bfi_score

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
    'MIN_LENGTH': 15,
    'DROP_NA': True,
    'DROP_MIN': True,
    'DROP_MAX': True,
    'DROP_NON_ENGLISH': True,       # remain only desc in english
    'DROP_NON_ENGLISH_WORDS': True  # remain only valid words in english
}

# POS properties
POS = {
    'VALID_POS': ['RBS', 'RB', 'RBR', 'JJ', 'JJR', 'JJS'],  # save only this POS in addition
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
    'TOP_K_WORDS': 30,           # present top words
    'SMOOTHING_FACTOR': 1.0,     # smoothing factor for calculate term contribution
    'NGRAM_RANGE': (1, 1),
    'VOCABULARY_METHOD': 'documents',    # 'documents', 'aggregation'
    'NORMALIZE_CONTRIBUTE': {
        'flag': False,
        'type': 'ratio'
    },
    'FIND_WORD_DESCRIPTION': {  # save top words with k description they appear in
        'flag': True,
        'k': 30
    }
}

personality_trait = ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']

balance_description = {
    'percentile_truncated': 95
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


