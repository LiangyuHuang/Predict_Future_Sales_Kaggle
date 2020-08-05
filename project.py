#!/usr/bin/python
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from itertools import product

def stacking(model, x_train, y_train, x_test, k_fold, num_k_fold):
    stk_train_data = np.zeros((x_train.shape[0],))
    stk_test = np.zeros((x_test.shape[0],))
    stk_test_kf = np.zeros((num_k_fold, x_test.shape[0]))
    i = 0
    for train_index, test_index in k_fold.split(x_train):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        model = model.fit(kf_x_train, kf_y_train)
        kf_x_test = x_train[test_index]
        stk_train_data[test_index] = model.predict(kf_x_test)
        stk_test_kf[i, :] = model.predict(x_test)
        i += 1
    stk_test[:] = stk_test_kf.mean(axis=0)
    return stk_train_data, stk_test

def stk_train(model, X_train, X_test, Y_train, kf, num_k_fold):
    new_x_train = np.zeros((X_train.shape[0], len(model)))
    new_x_test = np.zeros((X_test.shape[0], len(model)))
    i = 0
    for regression in model:
        new_x_train[:, i], new_x_test[:, i] = stacking(
            regression, X_train, Y_train, X_test, kf, num_k_fold)
        i += 1
    return new_x_train, new_x_test

def model_data_preprocessing(train_data,valid_data,test_data):
    # drop ID
    train_data = train_data.T[1:].T
    valid_data = valid_data.T[1:].T
    # get the test_id
    test_ID = list(range(0, test_data.shape[0]))
    # drop ID
    test_data = test_data.T[1:].T
    X_final_test = test_data.drop(['item_monthly'], axis=1).values
    # get the X data
    X_train_data = train_data.drop(['item_monthly'], axis=1).values
    X_valid_data = valid_data.drop(['item_monthly'], axis=1).values
    # get the Y data
    Y_train_data = train_data['item_monthly'].values.clip(0,20)
    Y_valid_data = valid_data['item_monthly'].values.clip(0,20)
    return test_ID,X_final_test,X_train_data,X_valid_data,Y_train_data,Y_valid_data

def test_models_performance(all_model,X_train,Y_train,X_valid,Y_valid):
    for i,regression in enumerate(all_model):
        tc = time.time()
        regression.fit(X_train, Y_train)
        Y_pred = regression.predict(X_valid)
        r_mse = sqrt(mean_squared_error(Y_valid, Y_pred))
        if i == 0:
            print(f'Lasso model\'s R_MSE :{r_mse:.4f}')
        elif i == 1:
            print(f'RandomForest model\'s R_MSE :{r_mse:.4f}')
        elif i == 2:
            print(f'XGBoost model\'s R_MSE :{r_mse:.4f}')
        elif i == 3:
            print(f'GradientBoosting model\'s R_MSE :{r_mse:.4f}')
        elif i == 4:
            print(f'LightGBM model\'s R_MSE :{r_mse:.4f}')
        elif i == 5:
            print(f'ENet model\'s R_MSE :{r_mse:.4f}')
        elif i == 6:
            print(f'KRR model\'s R_MSE :{r_mse:.4f}')
        tc = time.time() - tc
        print('time:', tc)

def test_stacking_performance(new_x_train,Y_train,new_x_test,Y_valid):
    tc = time.time()
    stacking = LinearRegression()
    stacking.fit(new_x_train, Y_train)
    Y_reg_pred = stacking.predict(new_x_test)
    r_mse = sqrt(mean_squared_error(Y_valid, Y_reg_pred))
    print(f'stacking model\'s RMSE :{r_mse:.4f}')
    tc = time.time() - tc
    print('time:', tc)

def creat_csv(test_ID,Y_final_pred):
    result = pd.DataFrame()
    result['ID'] = test_ID
    result['item_monthly'] = Y_final_pred
    # output to the current location
    current_path = os.path.abspath(__file__)
    location = (os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."), 'submission.csv'))
    result.to_csv(location, index=False)

def bulit_model_and_predict(data):

    train_data = data[data.date_block_num < 33]
    valid_data = data[data.date_block_num == 33]
    test_data = data[data.date_block_num == 34]

    # data processing before modeling
    test_ID, X_final_test, X_train, X_valid, Y_train, Y_valid = model_data_preprocessing(train_data,valid_data,test_data)

    # initialize k-fold
    num_k_fold = 5
    kf = KFold(n_splits=num_k_fold)

    # initialize model
    # lasso
    # lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=100))
    # RandomForest
    # useless in this project
    rf = RandomForestRegressor(n_estimators=10, criterion='mse',
                               max_features='sqrt', min_samples_leaf=15,
                               min_samples_split=10, max_depth=8)
    # XGBoost
    xgb = XGBRegressor(tree_method='hist', max_depth=8,
                       n_estimators=10, min_child_weight=300,
                       colsample_bytree=0.8, subsample=0.8,
                       eta=0.3, seed=42)
    # GradientBoosting
    # gdt = GradientBoostingRegressor(n_estimators=10, learning_rate=0.005,
    #                                 max_depth=8, max_features='sqrt',
    #                                 min_samples_leaf=10, min_samples_split=10,
    #                                 loss='huber', random_state=100)
    # lightGBM
    gbm = LGBMRegressor(objective='regression', num_leaves=8,
                        learning_rate=0.005, n_estimators=10,
                        max_bin=55, bagging_fraction=0.8,
                        bagging_freq=5, feature_fraction=0.2319,
                        feature_fraction_seed=9, bagging_seed=9,
                        min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
                        max_depth=8)
    # E_NET
    # e_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    # KRR
    # krr = KernelRidge(alpha=0.3, kernel='polynomial', degree=2, coef0=2.5)

    # used for stacking basic models
    # model = [xgb, gbm, krr]
    model = [xgb, gbm]
    # all model, for testing the score
    # all_model = [lasso, rf, xgb, gdt, gbm, e_net, krr]
    all_model = [gbm, xgb]

    # fitting and check the R_MSE of each model
    test_models_performance(all_model, X_train, Y_train, X_valid, Y_valid)

    new_x_train, new_x_test = stk_train(model, X_train, X_valid, Y_train, kf, num_k_fold)

    # check the R_MSE of ensemble learning
    test_stacking_performance(new_x_train, Y_train, new_x_test, Y_valid)

    # final predict
    final_x_train, final_x_test = stk_train(model, X_train, X_final_test, Y_train, kf, num_k_fold)
    ensemble = LinearRegression()
    ensemble.fit(final_x_train, Y_train)
    Y_final_pred = ensemble.predict(final_x_test)

    # creat csv
    creat_csv(test_ID,Y_final_pred)

def data_helper():

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
        total_df.append(np.array(list(product([i], df.shop_id.unique(), df.item_id.unique())), dtype='int16'))

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
    total_df = total_df.astype('int16')

    # # Create Lag Features and Mean-Encodings
    temp = sales.groupby(by=['shop_id', 'item_id', 'date_block_num']).agg(item_monthly=('item_cnt_day', sum))
    temp.columns = ['item_monthly']
    temp.reset_index(inplace=True)
    total_df = pd.merge(total_df, temp, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    total_df['item_monthly'] = total_df['item_monthly'].fillna(0).clip(0, 20).astype('float32')

    # Define functions
    def feature_helper(data_frame, lags, cols):
        for col in cols:
            tmp = data_frame[["date_block_num", "shop_id", "item_id", col]]
            for i in lags:
                tmp = tmp.copy()
                tmp.columns = ["date_block_num", "shop_id", "item_id", col + "_mean_"+str(i)]
                tmp.date_block_num = tmp.date_block_num + i
                tmp = tmp.astype('float32')
                data_frame = pd.merge(data_frame, tmp, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return data_frame

    def monthly_helper(data_frame, group_by_column, column_name, lags):
        temp_data = data_frame.groupby(group_by_column).agg({'item_monthly': ['mean']})
        temp_data.columns = column_name
        temp_data.reset_index(inplace=True)
        data_frame = pd.merge(data_frame, temp_data, on=group_by_column, how='left')
        data_frame = feature_helper(data_frame, lags, column_name)
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
    total_df['days'] = total_df['month'].map(days).astype('int16')
    total_df['years'] = total_df['date_block_num'].map(lambda x: scope(x)).astype('int16')
    total_df['month'] = (total_df['month'] + 1).astype('int16')  # fix month

    # The first month when one item is on sale
    item_shop_df = total_df.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
    item_df = total_df.groupby('item_id')['date_block_num'].transform('min')
    total_df['item_shop_sale_once'] = total_df['date_block_num'] - item_shop_df
    total_df['item_sale_once'] = total_df['date_block_num'] - item_df
    total_df = total_df[total_df["date_block_num"] > 3].fillna(0)

    print(total_df.info())
    print("data_processing output begin")
    total_df.to_pickle('data.pkl')
    print("data_processing end")

def main():
    # data processing part
    data_helper()

    # model part
    # when the data pre-processing is finished, change the following 2 rows
    # train_data = pd.read_csv('trainv4.csv').astype('int32')
    # test_data = pd.read_csv('testv4.csv').astype('int32')
    # valid_data = pd.read_csv('validationv4.csv').astype('int32')
    data = pd.read_pickle('data.pkl')
    bulit_model_and_predict(data)

if __name__ == '__main__':
    main()