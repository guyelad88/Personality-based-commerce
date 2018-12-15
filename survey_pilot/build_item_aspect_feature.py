from __future__ import print_function
import numpy as np
import pandas as pd
from utils.logger import Logger


class BuildItemAspectScore:

    def __init__(self, item_aspects_df, participant_df, purchase_history_df, valid_users_df, merge_df,
                 user_id_name_dict, aspect_feature):

        self.item_aspects_df = item_aspects_df
        self.participant_df = participant_df
        self.purchase_history_df = purchase_history_df
        self.valid_users_df = valid_users_df
        self.merge_df = merge_df
        self.user_id_name_dict = user_id_name_dict
        self.aspect_feature = aspect_feature          # feature to extract from item aspect
        self.name_user_id_dict = dict((v, k) for k, v in user_id_name_dict.iteritems())     # flip key value

    def add_aspect_features(self):
        self.add_feature_column()
        sum_row = 0

        grouped = self.purchase_history_df.groupby(['buyer_id'])  # group by how many each user bought
        for cur_user_name_id, group in grouped:
            cur_user_name = self.user_id_name_dict[float(list(group['buyer_id'])[0])]       # user_name_id
            # cur_user_name = self.user_id_name_dict[str(list(group['buyer_id'])[0])]  # user_name_id

            if cur_user_name in list(self.merge_df['eBay site user name']):

                # user row index
                cur_merge_idx_row = self.merge_df.index[self.merge_df['eBay site user name'] == cur_user_name].tolist()[0]
                cur_user_item_list = group['item_id'].tolist()
                df_aspect = self.item_aspects_df.loc[self.item_aspects_df['item_id'].isin(cur_user_item_list)]
                if df_aspect.shape[0] > 0:
                    num_product_aspect = len(df_aspect['item_id'].value_counts())   # number of products with aspects

                    # compute color item aspect ratio
                    if 'color_ratio' in self.aspect_feature:
                        # num_color_aspect = df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Color'].shape[0]
                        num_color_aspect = len(df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Color']['item_id'].value_counts())

                        self.merge_df.at[cur_merge_idx_row, 'color_ratio'] = \
                            min(1, float(num_color_aspect) / float(num_product_aspect))

                    # TODO add more colors as grey...
                    # compute colorful item aspect ratio (not white and black)
                    if 'colorful_ratio' in self.aspect_feature:
                        cur_color_aspect_df = df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Color']
                        num_color_aspect = max(cur_color_aspect_df.shape[0], 1)

                        cnt_colorful = 0     # count number of non-colorful items
                        for index, row in cur_color_aspect_df.iterrows():
                            if not pd.isnull(row['ASPCT_VLU_NM']):
                                if 'black' not in row['ASPCT_VLU_NM'].lower() and 'white' not in row['ASPCT_VLU_NM'].lower():
                                    cnt_colorful += 1

                        if num_color_aspect > 0:
                            self.merge_df.at[cur_merge_idx_row, 'colorful_ratio'] = \
                                min(1, float(cnt_colorful) / float(num_color_aspect))

                    # add brand ratio - product with defined brand name
                    if 'brand_ratio' in self.aspect_feature:
                        # num_brand_aspect = df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Brand']
                        num_brand_aspect = len(df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Brand']['item_id'].value_counts())

                        if num_product_aspect > 0:
                            self.merge_df.at[cur_merge_idx_row, 'brand_ratio'] = \
                                min(1, float(num_brand_aspect) / float(num_product_aspect))

                    # add unlabeled brand ratio
                    if 'brand_unlabeled_ratio' in self.aspect_feature:
                        cur_brand_aspect_df = df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Brand']
                        num_brand_aspect = cur_brand_aspect_df.shape[0]

                        cnt_un_brand = 0     # count number of unbranded items

                        for index, row in cur_brand_aspect_df.iterrows():
                            if not isinstance(row['ASPCT_VLU_NM'], basestring) and np.isnan(row['ASPCT_VLU_NM']):
                                continue
                            elif 'Unbrand' in row['ASPCT_VLU_NM']:
                                cnt_un_brand += 1

                        if num_brand_aspect > 0:
                            self.merge_df.at[cur_merge_idx_row, 'brand_unlabeled_ratio'] = \
                                min(1, float(cnt_un_brand) / float(num_brand_aspect))

                    # add country ratio
                    if 'country_ratio' in self.aspect_feature:
                        # num_country_aspect = df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Country/Region of Manufacture'].shape[0]
                        num_country_aspect = len(df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Country/Region of Manufacture']['item_id'].value_counts())

                        self.merge_df.at[cur_merge_idx_row, 'country_ratio'] = \
                            min(1, float(num_country_aspect) / float(num_product_aspect))

                    # add ratio of product with protection plan
                    if 'protection_ratio' in self.aspect_feature:
                        # num_protection_aspect = df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Protection Plan'].shape[0]

                        num_protection_aspect = len(df_aspect.loc[df_aspect['PRDCT_ASPCT_NM'] == 'Protection Plan']['item_id'].value_counts())
                        self.merge_df.at[cur_merge_idx_row, 'protection_ratio'] = min(1, float(num_protection_aspect) / float(num_product_aspect))

                sum_row = sum_row + df_aspect.shape[0]
                continue

        Logger.info('total aspect: ' + str(self.item_aspects_df.shape[0]))

    # create empty column of features
    def add_feature_column(self):
        if 'color_ratio' in self.aspect_feature:
            self.merge_df["color_ratio"] = 0.0
        if 'colorful_ratio' in self.aspect_feature:
            self.merge_df["colorful_ratio"] = 0.0
        if 'protection_ratio' in self.aspect_feature:
            self.merge_df["protection_ratio"] = 0.0
        if 'country_ratio' in self.aspect_feature:
            self.merge_df["country_ratio"] = 0.0
        if 'brand_ratio' in self.aspect_feature:
            self.merge_df["brand_ratio"] = 0.0
        if 'brand_unlabeled_ratio' in self.aspect_feature:
            self.merge_df["brand_unlabeled_ratio"] = 0.0

    def main(self):
        return

    if __name__ == '__main__':
        raise SystemExit('This file does not contain main')