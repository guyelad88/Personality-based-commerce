
# indicate which pre-processing steps needs to do
pre_processing_configs = {
    'remove_duplication': True,         # currently under extract_bfi_score
    'remove_fake_users': True,          # currently under extract_bfi_score

    'filter_description': True,
    'balance_description_per_user': True,

    'POS_filter': True,
    'Number_filter': False,             # TODO add this

    'analyze_PT_groups': True,          # add 'L'/'M'/'H' to each PT

    'calculate_kl': True
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
    'DROP_NON_ENGLISH': True
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
    'NGRAM_RANGE': (2, 2),
    'VOCABULARY_METHOD': 'documents',    # 'documents', 'aggregation'
    'TRAIT': 'extraversion',
    'NORMALIZE_CONTRIBUTE': {
        'flag': False,
        'type': 'ratio'
    },
    'FIND_WORD_DESCRIPTION': {  # save top words with k description they appear in
        'flag': True,
        'k': 10
    }
}

personality_trait = ['agreeableness', 'extraversion', 'openness', 'conscientiousness', 'neuroticism']

balance_description = {
    'percentile_truncated': 95
}
