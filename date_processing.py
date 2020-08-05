#!/usr/bin/env python
# coding: utf-8

def data_helper():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from itertools import product

    print("data_processing begin")
    sales = pd.read_csv('sales_train.csv')
    items = pd.read_csv('items.csv')
    items_categories = pd.read_csv('item_categories.csv')
    shops = pd.read_csv('shops.csv')
    test = pd.read_csv('test.csv')

    print("data_processing stage1 begin")
    sales = sales[sales['item_cnt_day'] > 0]
    sales = sales[sales['item_price'] > 0]

    # drop the data (item_cnt_day, item_price) which is too large
    sales = sales[sales['item_cnt_day'] < 800]
    sales = sales[sales['item_price'] < 70000]

    # reset index
    sales.reset_index(drop=True)

    # In here, although the open & close time shows the shops are different, shop_id of 0, 1, 11 are not train or test data. This means shop is the same, the merge is needed.
    # merge 0 with 57
    sales.loc[sales.shop_id == 0, 'shop_id'] = 57
    # 1 with 58
    sales.loc[sales.shop_id == 1, 'shop_id'] = 58
    # 10 with 11
    sales.loc[sales.shop_id == 11, 'shop_id'] = 10

    # After merging the shop id, the anaysis of shop name is needed.
    # In here, after using *'Google translate'*, the output shows the first part of shop_name is city name and the second part of shop_name is the kind of shop(shopping mall, store, etc.).
    # BTW, as the output shows, the "46 Сергиев Посад ТЦ "7Я"" needs change to '46 СергиевПосад ТЦ "7Я".
    # change "46 Сергиев Посад ТЦ "7Я"" to '46 СергиевПосад ТЦ "7Я"
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    # collect the city name and the kind of shop
    shops['shop_city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops['shop_kind'] = shops['shop_name'].str.split(' ').map(lambda x: x[1])

    # Using LabelEncoder
    # label encoder for shops
    shops['shop_city'] = LabelEncoder().fit_transform(shops['shop_city'])
    shops['shop_kind'] = LabelEncoder().fit_transform(shops['shop_kind'])
    shops = shops[['shop_id', 'shop_city', 'shop_kind']]

    print("data_processing stage2 begin")
    # Also, the check for items is needed.
    # Merging these two dataframe at frist
    items = pd.merge(items, items_categories, on='item_category_id')

    # classify the category of items and label encode it
    def check_item_category(x):
        if x in range(0, 8):
            return 'accessories'
        if x in range(10, 18):
            return 'consoles'
        if x in range(18, 32):
            return 'games'
        if x in range(32, 37):
            return 'payment_cards'
        if x in range(37, 42):
            return 'cinema'
        if x in range(42, 55):
            return 'books'
        if x in range(55, 61):
            return 'music'
        if x in range(61, 73):
            return 'gifts'
        if x in range(73, 79):
            return 'programs'
        if x in range(8, 81):
            return 'tickets'
        if x == 83:
            return 'batteries'
        if x == 9:
            return 'delivery'
        if x == 79:
            return 'office'
        if x in range(81, 83):
            return 'discs'

    items['item_category_name'] = items['item_category_id'].apply(lambda x: check_item_category(x))
    items['item_category_name'] = LabelEncoder().fit_transform(items['item_category_name'])
    items = items[['item_id', 'item_category_id', 'item_category_name']]

    print("data_processing stage3 begin: it may needs 10 min")
    # Finally, check test
    total_df = []
    for i in range(34):
        df = sales[sales.date_block_num == i]
        total_df.append(np.array(list(product([i], df.shop_id.unique(), df.item_id.unique())), dtype='int8'))

    total_df = pd.DataFrame(np.vstack(total_df), columns=['date_block_num', 'shop_id', 'item_id'])
    total_df.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)

    # deal with test data
    test = test[['shop_id', 'item_id']]
    test['date_block_num'] = 34

    # Merge the data
    total_df = pd.concat([total_df, test], ignore_index=True, keys=['date_block_num', 'shop_id', 'item_id'])
    total_df = pd.merge(total_df, shops, on=['shop_id'], how='left')
    total_df = pd.merge(total_df, items, on=['item_id'], how='left')
    total_df.fillna(0, inplace=True)
    total_df = total_df.astype('int8')

    # # Create Lag Features and Mean-Encodings
    temp = sales.groupby(by=['shop_id', 'item_id', 'date_block_num']).agg(item_monthly=('item_cnt_day', sum))
    temp.columns = ['item_monthly']
    temp.reset_index(inplace=True)
    total_df = pd.merge(total_df, temp, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    total_df['item_monthly'] = total_df['item_monthly'].fillna(0).clip(0, 20).astype('float16')

    # Define functions
    def feature_helper(data_frame, lags, cols):
        for col in cols:
            tmp = data_frame[["date_block_num", "shop_id", "item_id", col]]
            for i in lags:
                tmp = tmp.copy()
                tmp.columns = ["date_block_num", "shop_id", "item_id", col + "_mean_"+str(i)]
                tmp.date_block_num = tmp.date_block_num + i
                tmp = tmp.astype('float16')
                data_frame = pd.merge(data_frame, tmp, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return data_frame

    def monthly_helper(data_frame, group_by_column, column_name, lags):
        temp_data = data_frame.groupby(group_by_column).agg({'item_monthly': ['mean']})
        temp_data.columns = column_name
        temp_data.reset_index(inplace=True)
        data_frame = pd.merge(data_frame, temp_data, on=group_by_column, how='left')
        data_frame = feature_helper(data_frame, lags, column_name).astype('float16')
        data_frame.drop(column_name, axis=1, inplace=True)

    # item_cnt_month
    total_df = feature_helper(total_df, [1, 2, 3], ['item_monthly'])

    params = [
        # Monthly - item_cnt_month
        (['date_block_num'], ['month_mean'], [1, 2, 3]),
        # Monthly item - item_cnt_month
        (['date_block_num', 'item_id'], ['item_month_mean'], [1, 2, 3]),
        # Monthly shops - item_cnt_month
        (['date_block_num', 'shop_id'], ['shop_month_mean'], [1, 2, 3]),
        # Monthly item_category - item_cnt_month
        (['date_block_num', 'item_category_id'], ['item_category_month_mean'], [1, 2]),
        # Monthly shops item_category - item_cnt_month
        (['date_block_num', 'shop_id', 'item_category_id'], ['shops_item_category_id_month_mean'], [1, 2]),
        # Monthly shops item - item_cnt_month
        (['date_block_num', 'shop_id', 'item_id'], ['shops_item_month_mean'], [1, 2]),
        # Monthly shops subs_item category - item_cnt_month
        (['date_block_num', 'shop_id', 'item_category_name'], ['shops_item_category_month_mean'], [1, 2]),
        # Monthly shops_city - item_cnt_month
        (['date_block_num', 'shop_city'], ['shops_city_month_mean'], [1, 2])
    ]

    for param1, param2, param3 in params:
        monthly_helper(total_df, param1, param2, param3)

    def scope(x):
        if x in range(0, 12):
            return '13'
        if x in range(12, 25):
            return '14'
        if x in range(25, 35):
            return '15'

    print("data_processing stage4 begin")
    # days, month, and year features
    total_df['month'] = total_df['date_block_num'] % 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    total_df['days'] = total_df['month'].map(days).astype('int8')
    total_df['years'] = total_df['date_block_num'].map(lambda x: scope(x)).astype('int8')
    total_df['month'] = (total_df['month'] + 1).astype('int8')  # fix month

    # The first month when one item is on sale
    total_df['item_shop_sale_once'] = total_df['date_block_num'] - total_df.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
    total_df['item_sale_once'] = total_df['date_block_num'] - total_df.groupby('item_id')['date_block_num'].transform('min')
    total_df = total_df[total_df["date_block_num"] > 3].fillna(0)

    print(total_df.info())
    print("data_processing output begin")
    total_df.to_pickle('data.pkl')
    print("data_processing end")
