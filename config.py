
# indicate which pre-processing steps needs to do
pre_processing_configs = {
    'remove_duplication': True,         # currently under extract_bfi_score
    'remove_fake_users': True,          # currently under extract_bfi_score

    'POS_filter': False,
    'Number_filter': False,

    'filter_description': True,
    'balance_description_per_user': True
}

# indicate how to build the vocabularies (e.g. extroversion vs. introversion)
create_vocabularies = {
    'directory_output': '../results/vocabulary/',
    'vocabulary_method': 'documents',       # 'documents', 'aggregation'
    'verbose_flag': True,
    'split_method': 'traits',               # 'vertical', 'traits', 'traits_vertical'
    'gap_value': 0.1,                       # must be a float number between zero to one
    'personality_trait': 'extraversion',  # 'agreeableness' 'extraversion' 'openness' 'conscientiousness' 'neuroticism'
    'vertical': '',                         # 'Fashion'
}

# tune duplication validity and user name validity
extract_big_five_inventory_score = {
    'threshold': 0.8,                       # max diff in trait withing user with duplication
    'name_length_threshold': 5              # min name length
}

# determine how to filter descriptions
filter_description = {
    'MAX_LENGTH': 500,
    'MIN_LENGTH': 15,
    'DROP_NA': True,
    'DROP_MIN': True,
    'DROP_MAX': True,
    'DROP_NON_ENGLISH': True
}

# POS properties
POS = {
    'VALID_POS': ['RBS', 'RB', 'RBR', 'JJ', 'JJR', 'JJS']
}
