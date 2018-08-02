
# indicate which pre-processing steps needs to do
pre_processing_configs = {
    'remove_duplication': True,
    'remove_fake_users': True,

    'POS_filter': False,
    'Number_filter': False,

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