"""
'participant_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/clean_participant_695_2018-05-13 16:54:12.csv',
    'item_aspects_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/Item Aspects.csv',
    'purchase_history_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/personality_purchase_history.csv',
    'valid_users_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/personality_valid_users.csv',
"""
"""
'participant_file': '/Users/gelad/Personality-based-commerce/results/data/BFI_results/participant_bfi_score_clean_duplication/clean_participant_985_2018-12-07 15:44:21.csv',
"""

"""
common use: (until 8.12)
'participant_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/clean_participant_695_2018-05-13 16:54:12.csv',
'item_aspects_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/Item Aspects.csv',
'purchase_history_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/personality_purchase_history.csv',
'valid_users_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/personality_valid_users.csv',
"""


"""
from 8.12 threshold 5, distance: 0.8
# 'participant_file': '../results/data/BFI_results/participant_bfi_score_clean_duplication/clean_participant_1165_2018-12-07 22:26:53.csv',

from 9.12 threshold 6, distance: 0.5
# 'participant_file': '../results/data/BFI_results/participant_bfi_score_clean_duplication/clean_participant_1058_2018-12-08 20:20:50.csv',

from 9.12 - threshold 6, distance: 0.3
'participant_file': '../results/data/BFI_results/participant_bfi_score_clean_duplication/clean_participant_970_2018-12-08 21:45:57.csv',

from 9.12 - threshold 7, distance: 0.15
'participant_file': '../results/data/BFI_results/participant_bfi_score_clean_duplication/clean_participant_847_2018-12-08 23:08:05.csv',
'purchase_history_file': '/Users/gelad/Personality-based-commerce/data/participant_data/29_8_1018/mozart_run_16_stmt_1_0.csv',
'valid_users_file': '/Users/gelad/Personality-based-commerce/data/participant_data/29_8_1018/personality_valid_users.csv',
"""
predict_trait_configs = {

    'participant_file': '../results/data/BFI_results/participant_bfi_score_clean_duplication/clean_participant_847_2018-12-08 23:08:05.csv',
    'purchase_history_file': '/Users/gelad/Personality-based-commerce/data/participant_data/29_8_1018/mozart_run_16_stmt_1_0.csv',
    'valid_users_file': '/Users/gelad/Personality-based-commerce/data/participant_data/29_8_1018/personality_valid_users.csv',

    'item_aspects_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/Item Aspects.csv',
    'dir_analyze_name': '/Users/gelad/Personality-based-commerce/results/BFI_results/analyze_CF/',
    'dir_logistic_results': '/Users/gelad/Personality-based-commerce/results/BFI_results/',

    'dict_feature_flag': {
        'time_purchase_ratio_feature_flag': True,
        'time_purchase_meta_feature_flag': True,
        'vertical_ratio_feature_flag': True,
        'purchase_percentile_feature_flag': True,
        'user_meta_feature_flag': True,
        'aspect_feature_flag': False
    },

    'predefined_data_set_flag': False,
    'predefined_data_set_path': '/Users/gelad/Personality-based-commerce/results/BFI_results/pre_defined_df/shape=227_58_time=_2018-12-06 15:02:03.csv',

    'model_method': 'logistic',
    'classifier_type': 'xgb',
    'split_bool': True,
    'user_type': 'cf',
    'l_limit': 0.3,
    'h_limit': 0.7,

    'k_best_feature_flag': True,
    'k_best_list': [12, 15],

    'threshold_list': [20, 30, 40],
    'penalty': ['l1'],
    'C': 1,
    'xgb_c': [3, 2, 1, 0.3, 0.01, 0.001],
    'xgb_eta': [0.3, 0.01, 0.001],
    'xgb_max_depth': [1, 2, 3, 5, 7, 9],
    'bool_slice_gap_percentile': True,
    'xgb_n_estimators': 500,
    'xgb_subsample': 1,
    'xgb_colsample_bytree': 1
}

"""
    'k_best_feature_flag': True,
    'k_best_list': [12, 15],

    'threshold_list': [20, 30, 40],
    'penalty': ['l1'],
    'C': 1,
    'xgb_c': [1, 0.3, 0.01, 0.001],
    'xgb_eta': [0.3, 0.01, 0.001],
    'xgb_max_depth': [1, 2, 3, 5, 7],
    'bool_slice_gap_percentile': True,
    'xgb_n_estimators': 500,
    'xgb_subsample': 1,
    'xgb_colsample_bytree': 1
"""
"""
    'k_best_feature_flag': True,
    'k_best_list': [12, 15],

    'threshold_list': [20, 30, 40],
    'penalty': ['l1'],
    'C': 1,
    'xgb_c': [1, 0.3, 0.01, 0.001],
    'xgb_eta': [0.3, 0.01, 0.001],
    'xgb_max_depth': [1, 2, 3, 5, 7],
    'bool_slice_gap_percentile': True,
    'xgb_n_estimators': 500,
    'xgb_subsample': 1,
    'xgb_colsample_bytree': 1
"""
"""
'threshold_list': [20, 30, 40],
    'penalty': ['l1'],
    'C': 1,
    'xgb_c': [0.3, 0.01],
    'xgb_eta': [0.3, 0.01, 0.001],
    'xgb_max_depth': [2, 3, 5, 7],
    'bool_slice_gap_percentile': True,
    'xgb_n_estimators': 500,
    'xgb_subsample': 1,
    'xgb_colsample_bytree': 1
"""
"""
    'k_best_feature_flag': False,
    'k_best_list': [30],
    'threshold_list': [30, 40],
    'penalty': ['l1'],
    'C': 1,
    'xgb_c': [0.001, 0.01, 0.3],
    'xgb_eta': [0.5, 0.1, 0.01],
    'xgb_max_depth': [1, 2, 3, 5],
    'bool_slice_gap_percentile': True
    'xgb_n_estimators': 1000,
    'xgb_subsample': 1, 
    'xgb_colsample_bytree': 1
"""

bfi_test_information = {
    'question_openness': [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
    'question_conscientiousness': [3, 8, 13, 18, 23, 28, 33, 43],
    'question_extraversion': [1, 6, 11, 16, 21, 26, 31, 36],
    'question_agreeableness': [2, 7, 12, 17, 22, 27, 32, 37, 42],
    'question_neuroticism': [4, 9, 14, 19, 24, 29, 34, 39]
}

feature_data_set = {
        'pearson_relevant_feature': ['Age', 'openness_percentile',
                   'conscientiousness_percentile', 'extraversion_percentile', 'agreeableness_percentile',
                   'neuroticism_percentile', 'number_purchase', 'Electronics_ratio', 'Fashion_ratio',
                   'Home & Garden_ratio', 'Collectibles_ratio', 'Lifestyle_ratio', 'Parts & Accessories_ratio',
                   'Business & Industrial_ratio', 'Media_ratio'],

        'lr_y_feature': ['agreeableness_trait', 'extraversion_trait', 'neuroticism_trait', 'conscientiousness_trait', 'openness_trait'],
        'lr_y_logistic_feature': ['openness_group', 'conscientiousness_group', 'extraversion_group','agreeableness_group', 'neuroticism_group'],
        'lr_y_linear_feature': ['openness_group', 'conscientiousness_group', 'extraversion_group', 'agreeableness_group', 'neuroticism_group'],
        'trait_percentile': ['openness_percentile', 'conscientiousness_percentile', 'extraversion_percentile', 'agreeableness_percentile', 'neuroticism_percentile'],

        'map_dict_percentile_group': {
            'extraversion_group': 'extraversion_percentile',
            'openness_group': 'openness_percentile',
            'conscientiousness_group': 'conscientiousness_percentile',
            'agreeableness_group': 'agreeableness_percentile',
            'neuroticism_group': 'neuroticism_percentile'

        },
        'time_purchase_ratio_feature': ['day_ratio', 'evening_ratio', 'night_ratio', 'weekend_ratio'],
        'time_purchase_meta_feature': ['first_purchase', 'last_purchase', 'tempo_purchase'],

        'vertical_ratio_feature': [
            'Electronics_ratio', 'Fashion_ratio', 'Home & Garden_ratio', 'Collectibles_ratio','Lifestyle_ratio',
            'Parts & Accessories_ratio', 'Business & Industrial_ratio', 'Media_ratio'],

        'purchase_price_feature': [
            'median_purchase_price', 'q1_purchase_price', 'q3_purchase_price', 'min_purchase_price',
            'max_purchase_price'],

        'purchase_percentile_feature': ['median_purchase_price_percentile', 'q1_purchase_price_percentile',
                                        'q3_purchase_price_percentile', 'min_purchase_price_percentile',
                                        'max_purchase_price_percentile'],

        'user_meta_feature': ['Age', 'gender', 'number_purchase'],
        'aspect_feature': ['color_ratio', 'colorful_ratio', 'protection_ratio', 'country_ratio', 'brand_ratio',
                           'brand_unlabeled_ratio']
}
