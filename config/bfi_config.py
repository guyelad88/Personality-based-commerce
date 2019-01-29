predict_trait_configs = {

    'participant_file': '../results/data/BFI_results/participant_bfi_score_clean_duplication/clean_participant_847_2018-12-08 23:08:05.csv',
    'purchase_history_file': '/Users/gelad/Personality-based-commerce/data/participant_data/29_8_1018/mozart_run_16_stmt_1_0.csv',
    'valid_users_file': '/Users/gelad/Personality-based-commerce/data/participant_data/29_8_1018/personality_valid_users.csv',

    'item_aspects_file': '/Users/gelad/Personality-based-commerce/data/participant_data/1425 users input/Item Aspects.csv',
    'dir_analyze_name': '/Users/gelad/Personality-based-commerce/results/BFI_results/analyze_CF/',
    'dir_logistic_results': '/Users/gelad/Personality-based-commerce/results/BFI_results/',

    'title_corpus': '/Users/gelad/Personality-based-commerce/data/participant_data/purchase_data/titles_corpus_710.csv',
    'description_corpus': '/Users/gelad/Personality-based-commerce/data/participant_data/purchase_data/descriptions_corpus_438.csv',
    'color_corpus': '/Users/gelad/Personality-based-commerce/data/participant_data/purchase_data/color_groups_corpus_413.csv',

    'dict_feature_flag': {
        'purchase_percentile_feature_flag': True,
        'time_purchase_ratio_feature_flag': True,
        'time_purchase_meta_feature_flag': True,
        'user_meta_feature_flag': True,
        'vertical_ratio_feature_flag': True,
        'meta_category_feature_flag': True,

        'title_feature_flag': True,
        'descriptions_feature_flag': False,

        'color_feature_flag': False,

        'aspect_feature_flag': False
    },

    # load pre-defined data set - problem due to slice already min purchase amount etc.

    'predefined_data_set_path': '/Users/gelad/Personality-based-commerce/results/BFI_results/pre_defined_df/shape=227_58_time=_2018-12-06 15:02:03.csv',
    'predefined_data_set_flag': False,

    'model_method': 'logistic',
    'bool_slice_gap_percentile': True,
    'split_bool': True,

    'k_rand': [199, 200],

    'l_limit': 0.3,
    'h_limit': 0.7,
    'num_splits': 5,

    'k_best_feature_flag': True,
    'k_best_list': [50, 100, 200],           # , 100, 250],  # [100, 200, 50],  # [100, 50, 300], [30, 15, 10]
    'classifier_type': 'lr',                          # 'xgb', lr
    'user_type': 'all',                                 # 'cf', 'all'
    'threshold_list': [20, 30, 40],  # 20               # select 0 to save all users (on predefined df)
    'penalty': ['l2', 'l1'],
    'C': 1,
    # 'xgb_c': [1, 0.1, 5, 3, 100, 20, 2, 600, 500, 400, 300, 200, 50, 30, 17, 15, 14, 12, 10, 8, 6, 1000],
    'xgb_c': [0.1, 1, 5, 10, 20],  # [0.1, 1, 5],
    'xgb_eta': [0.1],     # [0.3, 0.01],    # 4, 0.3, 0.01, 0.001], # 0.3, 0.01, 0.001],
    'xgb_max_depth': [5],  # , 5     #, 3, 5, 7, 1], # [9, 2, 3, 5, 7],

    'xgb_n_estimators': '',                # un-relevant - randomize value during inside
    'xgb_subsample': '',                     # un-relevant - randomize value during inside
    'xgb_colsample_bytree': '',               # un-relevant - randomize value during inside

    'min_df': 60,
    'max_textual_features': 300,

    'embedding_dim': 300,
    'embedding_limit': 100000,
    'embedding_type': 'fasttext',          # 'glove': 50,100,200,300 'ft_amazon': 300, fasttext: 300

    'dict_vec': {
        'max_features': 1000,                   # 30000 number of added features
        'ngram_range': [1, 2],
        'stop_words': 'english',
        'min_df': 3,
        'max_df': 0.3,
        'norm': 'l2',
        'missing_val': 'max_idf',               # 'max_idf', 'avg_idf', 'zero' - relevant in embedding
        'vec_type': 'vec_tfidf_embeds'
        # 'vec_type': 'vec_tfidf_embeds'        # by pertrained embddeing dim
        # 'vec_type': 'vec_count'               # by 'dict_vec'['max_features']
        # 'vec_type': 'vec_avg_embds'               # by 'dict_vec'['max_features']
        # 'vec_type': 'vec_avg_embds'           # by pertrained embddeing dim
    },
}

bfi_test_information = {
    'question_openness': [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
    'question_conscientiousness': [3, 8, 13, 18, 23, 28, 33, 43],
    'question_extraversion': [1, 6, 11, 16, 21, 26, 31, 36],
    'question_agreeableness': [2, 7, 12, 17, 22, 27, 32, 37, 42],
    'question_neuroticism': [4, 9, 14, 19, 24, 29, 34, 39]
}

black_list = ['123ebay','1768064','9285414','h_vivic_vpxwxxtjql','l0rd0ct0d0rk','pug.the.wizard','normality_is_an_illusion','fms*play','mortalis_123','mysss123','redzeb123']

feature_data_set = {

        'lr_y_logistic_feature': ['openness_group', 'conscientiousness_group', 'extraversion_group','agreeableness_group', 'neuroticism_group'],
        # all
        # 'lr_y_logistic_feature': ['openness_group', 'conscientiousness_group', 'extraversion_group','agreeableness_group', 'neuroticism_group'],

        'pearson_relevant_feature': ['Age', 'openness_percentile',
                   'conscientiousness_percentile', 'extraversion_percentile', 'agreeableness_percentile',
                   'neuroticism_percentile', 'number_purchase', 'Electronics_ratio', 'Fashion_ratio',
                   'Home & Garden_ratio', 'Collectibles_ratio', 'Lifestyle_ratio', 'Parts & Accessories_ratio',
                   'Business & Industrial_ratio', 'Media_ratio'],

        'lr_y_feature': ['agreeableness_trait', 'extraversion_trait', 'neuroticism_trait', 'conscientiousness_trait', 'openness_trait'],
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
        'time_purchase_meta_feature': ['first_purchase', 'last_purchase', 'tempo_purchase', 'number_purchase'],

        'vertical_ratio_feature': [
            'Electronics_ratio', 'Fashion_ratio', 'Home & Garden_ratio', 'Collectibles_ratio','Lifestyle_ratio',
            'Parts & Accessories_ratio', 'Business & Industrial_ratio', 'Media_ratio'],

        'categ_threshold': 0,
        'meta_category_feature': ['Collectibles', 'Books, Comics & Magazines', 'Jewelry & Watches', 'Health & Beauty', 'Home & Garden', 'Cell Phones & Accessories', 'Clothing, Shoes & Accessories', 'Computers/Tablets & Networking', 'Toys & Hobbies', 'Sporting Goods', 'Collectibles_meta_cat_ratio', 'Crafts_meta_cat_ratio', 'Pet Supplies_meta_cat_ratio', 'Coins & Paper Money_meta_cat_ratio', 'Books, Comics & Magazines_meta_cat_ratio', 'Vehicle Parts & Accessories_meta_cat_ratio', 'Events Tickets_meta_cat_ratio', 'Jewelry & Watches_meta_cat_ratio', 'Health & Beauty_meta_cat_ratio', 'Video Games & Consoles_meta_cat_ratio', 'Business & Industrial_meta_cat_ratio', 'Musical Instruments & Gear_meta_cat_ratio', 'Consumer Electronics_meta_cat_ratio', 'Art_meta_cat_ratio', 'Cameras & Photo_meta_cat_ratio', 'Baby_meta_cat_ratio', 'Entertainment Memorabilia_meta_cat_ratio', 'Home & Garden_meta_cat_ratio', 'Cell Phones & Accessories_meta_cat_ratio', 'Travel_meta_cat_ratio', 'Clothing, Shoes & Accessories_meta_cat_ratio', 'Garden & Patio_meta_cat_ratio', 'Computers/Tablets & Networking_meta_cat_ratio', 'Toys & Hobbies_meta_cat_ratio', 'DVDs & Movies_meta_cat_ratio', 'Music_meta_cat_ratio', 'Sports Mem, Cards & Fan Shop_meta_cat_ratio', 'Everything Else_meta_cat_ratio', 'Haushaltsger\xc3\xa4te_meta_cat_ratio', 'Pottery & Glass_meta_cat_ratio', 'Gift Cards & Coupons_meta_cat_ratio', 'Dolls & Bears_meta_cat_ratio', 'eBay Motors_meta_cat_ratio', 'Antiques_meta_cat_ratio', 'Sporting Goods_meta_cat_ratio'],

        'purchase_price_feature': [
            'median_purchase_price', 'q1_purchase_price', 'q3_purchase_price', 'min_purchase_price',
            'max_purchase_price'],

        'purchase_percentile_feature': ['median_purchase_price_percentile', 'q1_purchase_price_percentile',
                                        'q3_purchase_price_percentile', 'min_purchase_price_percentile',
                                        'max_purchase_price_percentile'],

        'user_meta_feature': ['Age', 'gender'],
        'aspect_feature': ['color_ratio', 'colorful_ratio', 'protection_ratio', 'country_ratio', 'brand_ratio',
                           'brand_unlabeled_ratio'],

        'color_feature': ['copper',
                         'indigo',
                         'gold',
                         'maroon',
                         'pink',
                         'violet',
                         'navy',
                         'magenta',
                         'mint',
                         'lime',
                         'blue',
                         'purple',
                         'black',
                         'aquamarine',
                         'white',
                         'cream',
                         'red',
                         'brown',
                         'turquoise',
                         'orange',
                         'yellow',
                         'olive',
                         'cyan',
                         'silver',
                         'bronze',
                         'gray',
                         'grey',
                         'green',
                         'beige',
                         'teal',
                         'color_ratio',
                         'copper_groups',
                         'indigo_groups',
                         'gold_groups',
                         'maroon_groups',
                         'pink_groups',
                         'violet_groups',
                         'navy_groups',
                         'magenta_groups',
                         'mint_groups',
                         'lime_groups',
                         'blue_groups',
                         'purple_groups',
                         'black_groups',
                         'aquamarine_groups',
                         'white_groups',
                         'cream_groups',
                         'red_groups',
                         'brown_groups',
                         'turquoise_groups',
                         'orange_groups',
                         'yellow_groups',
                         'olive_groups',
                         'cyan_groups',
                         'silver_groups',
                         'bronze_groups',
                         'gray_groups',
                         'grey_groups',
                         'green_groups',
                         'beige_groups',
                         'teal_groups',
                         'color_ratio_groups']
}

# all meta_category (without any threshold)
"""
'meta_category_feature': [
            'Collectibles',
             'Crafts',
             'Stamps',
             'Pet Supplies',
             'Coins & Paper Money',
             'Books, Comics & Magazines',
             'Nautica e imbarcazioni',
             'Business',
             'Half Video Games',
             'Home Entertainment',
             'Industrial',
             'Vehicle Parts & Accessories',
             'Events Tickets',
             'Jewelry & Watches',
             'Health & Beauty',
             'Video Games & Consoles',
             'Business & Industrial',
             'Modellbau',
             'Musical Instruments & Gear',
             '\xe5\xa5\xb3\xe5\xa3\xab\xe7\xae\xb1\xe5\x8c\x85/\xe9\x9e\x8b\xe5\xb8\xbd/\xe9\x85\x8d\xe4\xbb\xb6',
             'Consumer Electronics',
             'Art',
             'Cameras & Photo',
             'Baby',
             'Entertainment Memorabilia',
             'Real Estate',
             'Home & Garden',
             'Cell Phones & Accessories',
             'Travel',
             'Clothing, Shoes & Accessories',
             'Garden & Patio',
             'Specialty Services',
             'Fumetti',
             'Laptops & Computer Peripherals',
             'Wholesale & Job Lots',
             'Antiquit\xc3\xa4ten & Kunst',
             'Auto & Motorrad: Fahrzeuge',
             'Computers/Tablets & Networking',
             'Mobile Accessories',
             'Tickets, Travel',
             'B\xc3\xbcro & Schreibwaren',
             'Vini, caff\xc3\xa8 e gastronomia',
             'Toys & Hobbies',
             'DVDs & Movies',
             'Music',
             'Sports Mem, Cards & Fan Shop',
             'Everything Else',
             'Alcohol & Food',
             'Haushaltsger\xc3\xa4te',
             'Pottery & Glass',
             'Gift Cards & Coupons',
             'Half Music',
             'Half Movies',
             'Dolls & Bears',
             'Half Books',
             'eBay Motors',
             'Antiques',
             'Heimwerker',
             'Watches',
             'Tools , Hardware & Electricals',
             'Sporting Goods',
             'Collectibles_meta_cat_ratio',
             'Crafts_meta_cat_ratio',
             'Stamps_meta_cat_ratio',
             'Pet Supplies_meta_cat_ratio',
             'Coins & Paper Money_meta_cat_ratio',
             'Books, Comics & Magazines_meta_cat_ratio',
             'Nautica e imbarcazioni_meta_cat_ratio',
             'Business_meta_cat_ratio',
             'Half Video Games_meta_cat_ratio',
             'Home Entertainment_meta_cat_ratio',
             'Industrial_meta_cat_ratio',
             'Vehicle Parts & Accessories_meta_cat_ratio',
             'Events Tickets_meta_cat_ratio',
             'Jewelry & Watches_meta_cat_ratio',
             'Health & Beauty_meta_cat_ratio',
             'Video Games & Consoles_meta_cat_ratio',
             'Business & Industrial_meta_cat_ratio',
             'Modellbau_meta_cat_ratio',
             'Musical Instruments & Gear_meta_cat_ratio',
             '\xe5\xa5\xb3\xe5\xa3\xab\xe7\xae\xb1\xe5\x8c\x85/\xe9\x9e\x8b\xe5\xb8\xbd/\xe9\x85\x8d\xe4\xbb\xb6_meta_cat_ratio',
             'Consumer Electronics_meta_cat_ratio',
             'Art_meta_cat_ratio',
             'Cameras & Photo_meta_cat_ratio',
             'Baby_meta_cat_ratio',
             'Entertainment Memorabilia_meta_cat_ratio',
             'Real Estate_meta_cat_ratio',
             'Home & Garden_meta_cat_ratio',
             'Cell Phones & Accessories_meta_cat_ratio',
             'Travel_meta_cat_ratio',
             'Clothing, Shoes & Accessories_meta_cat_ratio',
             'Garden & Patio_meta_cat_ratio',
             'Specialty Services_meta_cat_ratio',
             'Fumetti_meta_cat_ratio',
             'Laptops & Computer Peripherals_meta_cat_ratio',
             'Wholesale & Job Lots_meta_cat_ratio',
             'Antiquit\xc3\xa4ten & Kunst_meta_cat_ratio',
             'Auto & Motorrad: Fahrzeuge_meta_cat_ratio',
             'Computers/Tablets & Networking_meta_cat_ratio',
             'Mobile Accessories_meta_cat_ratio',
             'Tickets, Travel_meta_cat_ratio',
             'B\xc3\xbcro & Schreibwaren_meta_cat_ratio',
             'Vini, caff\xc3\xa8 e gastronomia_meta_cat_ratio',
             'Toys & Hobbies_meta_cat_ratio',
             'DVDs & Movies_meta_cat_ratio',
             'Music_meta_cat_ratio',
             'Sports Mem, Cards & Fan Shop_meta_cat_ratio',
             'Everything Else_meta_cat_ratio',
             'Alcohol & Food_meta_cat_ratio',
             'Haushaltsger\xc3\xa4te_meta_cat_ratio',
             'Pottery & Glass_meta_cat_ratio',
             'Gift Cards & Coupons_meta_cat_ratio',
             'Half Music_meta_cat_ratio',
             'Half Movies_meta_cat_ratio',
             'Dolls & Bears_meta_cat_ratio',
             'Half Books_meta_cat_ratio',
             'eBay Motors_meta_cat_ratio',
             'Antiques_meta_cat_ratio',
             'Heimwerker_meta_cat_ratio',
             'Watches_meta_cat_ratio',
             'Tools , Hardware & Electricals_meta_cat_ratio',
             'Sporting Goods_meta_cat_ratio',
        ],
"""

# above 0.1 threshold unique values
"""
['Clothing, Shoes & Accessories', 'Collectibles_meta_cat_ratio', 'Crafts_meta_cat_ratio', 'Pet Supplies_meta_cat_ratio', 'Books, Comics & Magazines_meta_cat_ratio', 
'Vehicle Parts & Accessories_meta_cat_ratio', 'Jewelry & Watches_meta_cat_ratio', 'Health & Beauty_meta_cat_ratio', 'Video Games & Consoles_meta_cat_ratio', 
'Business & Industrial_meta_cat_ratio', 'Musical Instruments & Gear_meta_cat_ratio', 'Consumer Electronics_meta_cat_ratio', 'Cameras & Photo_meta_cat_ratio', 
'Baby_meta_cat_ratio', 'Home & Garden_meta_cat_ratio', 'Cell Phones & Accessories_meta_cat_ratio', 'Clothing, Shoes & Accessories_meta_cat_ratio', 
'Computers/Tablets & Networking_meta_cat_ratio', 'Toys & Hobbies_meta_cat_ratio', 'DVDs & Movies_meta_cat_ratio', 'Music_meta_cat_ratio', 
'Sports Mem, Cards & Fan Shop_meta_cat_ratio', 'Everything Else_meta_cat_ratio', 'eBay Motors_meta_cat_ratio', 'Sporting Goods_meta_cat_ratio']
"""

# above 0.05 threshold unique values
"""
['Collectibles', 'Books, Comics & Magazines', 'Jewelry & Watches', 'Health & Beauty', 'Home & Garden', 'Cell Phones & Accessories', 'Clothing, Shoes & Accessories', 'Computers/Tablets & Networking', 'Toys & Hobbies', 'Sporting Goods', 'Collectibles_meta_cat_ratio', 'Crafts_meta_cat_ratio', 'Pet Supplies_meta_cat_ratio', 'Coins & Paper Money_meta_cat_ratio', 'Books, Comics & Magazines_meta_cat_ratio', 'Vehicle Parts & Accessories_meta_cat_ratio', 'Events Tickets_meta_cat_ratio', 'Jewelry & Watches_meta_cat_ratio', 'Health & Beauty_meta_cat_ratio', 'Video Games & Consoles_meta_cat_ratio', 'Business & Industrial_meta_cat_ratio', 'Musical Instruments & Gear_meta_cat_ratio', 'Consumer Electronics_meta_cat_ratio', 'Art_meta_cat_ratio', 'Cameras & Photo_meta_cat_ratio', 'Baby_meta_cat_ratio', 'Entertainment Memorabilia_meta_cat_ratio', 'Home & Garden_meta_cat_ratio', 'Cell Phones & Accessories_meta_cat_ratio', 'Travel_meta_cat_ratio', 'Clothing, Shoes & Accessories_meta_cat_ratio', 'Garden & Patio_meta_cat_ratio', 'Computers/Tablets & Networking_meta_cat_ratio', 'Toys & Hobbies_meta_cat_ratio', 'DVDs & Movies_meta_cat_ratio', 'Music_meta_cat_ratio', 'Sports Mem, Cards & Fan Shop_meta_cat_ratio', 'Everything Else_meta_cat_ratio', 'Haushaltsger\xc3\xa4te_meta_cat_ratio', 'Pottery & Glass_meta_cat_ratio', 'Gift Cards & Coupons_meta_cat_ratio', 'Dolls & Bears_meta_cat_ratio', 'eBay Motors_meta_cat_ratio', 'Antiques_meta_cat_ratio', 'Sporting Goods_meta_cat_ratio']
"""

# at least 500 purchases
"""
# all category with at least 500 purchases
        'meta_category_feature': ['Collectibles', 'Crafts', 'Stamps', 'Pet Supplies', 'Books, Comics & Magazines', 'Vehicle Parts & Accessories', 'Baby', 'Events Tickets', 'Jewelry & Watches',
                                  'Health & Beauty', 'Business & Industrial', 'eBay Motors', 'Consumer Electronics', 'Entertainment Memorabilia', 'Home & Garden', 'Cell Phones & Accessories',
                                  'Clothing, Shoes & Accessories', 'Music', 'Computers/Tablets & Networking', 'Toys & Hobbies', 'DVDs & Movies', 'Video Games & Consoles',
                                  'Sports Mem, Cards & Fan Shop', 'Pottery & Glass', 'Musical Instruments & Gear', 'Coins & Paper Money', 'Cameras & Photo', 'Sporting Goods',
                                  'Collectibles_meta_cat_ratio', 'Crafts_meta_cat_ratio', 'Stamps_meta_cat_ratio', 'Pet Supplies_meta_cat_ratio',
                                  'Books, Comics & Magazines_meta_cat_ratio', 'Vehicle Parts & Accessories_meta_cat_ratio', 'Baby_meta_cat_ratio',
                                  'Events Tickets_meta_cat_ratio', 'Jewelry & Watches_meta_cat_ratio', 'Health & Beauty_meta_cat_ratio',
                                  'Business & Industrial_meta_cat_ratio', 'eBay Motors_meta_cat_ratio', 'Consumer Electronics_meta_cat_ratio',
                                  'Entertainment Memorabilia_meta_cat_ratio', 'Home & Garden_meta_cat_ratio', 'Cell Phones & Accessories_meta_cat_ratio',
                                  'Clothing, Shoes & Accessories_meta_cat_ratio', 'Music_meta_cat_ratio', 'Computers/Tablets & Networking_meta_cat_ratio',
                                  'Toys & Hobbies_meta_cat_ratio', 'DVDs & Movies_meta_cat_ratio', 'Video Games & Consoles_meta_cat_ratio',
                                  'Sports Mem, Cards & Fan Shop_meta_cat_ratio', 'Pottery & Glass_meta_cat_ratio',
                                  'Musical Instruments & Gear_meta_cat_ratio', 'Coins & Paper Money_meta_cat_ratio',
                                  'Cameras & Photo_meta_cat_ratio', 'Sporting Goods_meta_cat_ratio'],
"""