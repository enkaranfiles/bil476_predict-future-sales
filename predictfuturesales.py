import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

# from itertools import product
# from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

xgboost_try = 0
decision_tree_try = 0
random_forest_try = 0
lightGBM_try = 1


# =============================================================================
# The lines where we processed our data
# =============================================================================
def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df


items = pd.read_csv(r'competitive-data-science-predict-future-sales\items.csv')
shops = pd.read_csv(r'competitive-data-science-predict-future-sales\shops.csv')
cats = pd.read_csv(r'competitive-data-science-predict-future-sales\item_categories.csv')
train = pd.read_csv(r'competitive-data-science-predict-future-sales\sales_train.csv')
test = pd.read_csv(r'competitive-data-science-predict-future-sales\test.csv').set_index('ID')

train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 1001]

median = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num == 4) & (
            train.item_price > 0)].item_price.median()
train.loc[train.item_price < 0, 'item_price'] = median

train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+',
                                                                                                          '').str.strip()
shops['city'] = shops['shop_name'].str.partition(' ')[0]
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops['shop_type'] = shops['shop_name'].apply(lambda
                                                  x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
shops['shop_type'] = LabelEncoder().fit_transform(shops['shop_type'])
shops = shops[['shop_id', 'city_code', 'shop_type']]

cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
# if subtype is nan then type
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id', 'type_code', 'subtype_code']]
items.drop(['item_name'], axis=1, inplace=True)

matrix = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)
