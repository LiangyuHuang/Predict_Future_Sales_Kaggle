#!/usr/bin/python
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import pandas as pd

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

if __name__ == '__main__':
    # data processing

    # model
    # initialize k-fold
    num_k_fold = 5
    kf = KFold(n_splits=num_k_fold)

    # initialize model
    # lasso
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0003, random_state=100))
    # RandomForest
    # useless in this project
    rf = RandomForestRegressor(n_estimators=3000,
                               criterion='mse',
                               max_features='sqrt',
                               min_samples_leaf=15,
                               min_samples_split=10,
                               max_depth=8)
    # XGBoost
    xgb = XGBRegressor(tree_method='hist',
                       max_depth=8,
                       n_estimators=3000,
                       min_child_weight=300,
                       colsample_bytree=0.8,
                       subsample=0.8,
                       eta=0.3,
                       seed=42)
    # GradientBoosting
    gdt = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.005,
                                    max_depth=8, max_features='sqrt',
                                    min_samples_leaf=10, min_samples_split=10,
                                    loss='huber', random_state=100)
    # lightGBM
    gbm = LGBMRegressor(objective='regression', num_leaves=8,
                        learning_rate=0.005, n_estimators=3000,
                        max_bin=55, bagging_fraction=0.8,
                        bagging_freq=5, feature_fraction=0.2319,
                        feature_fraction_seed=9, bagging_seed=9,
                        min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
                        max_depth=8
                        )
    e_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

    krr = KernelRidge(alpha=0.3, kernel='polynomial', degree=2, coef0=2.5)

    # used for stacking basic models
    model = [xgb, gbm, krr]
    # all model, for testing the score
    all_model = [lasso, rf, xgb, gdt, gbm, e_net, krr]
    # import data
    train_data = pd.read_csv('Data/train_finalv2_1.csv')
    # drop ID
    train_data = train_data.T[1:].T
    # print (train_data)
    test_data = pd.read_csv('Data/test_fianlv2_1.csv')
    # print(test_data)
    # get the test_id
    test_ID = np.array(test_data['Unnamed: 0'].values)
    # print(test_ID)
    # drop ID
    test_data = test_data.T[1:].T
    # print(test_data)
    X_final_test = test_data.values
    # print(X_final_test)

    X_data = train_data.drop(axis=1, columns='SalePrice', inplace=False).values

    Y_data = train_data['SalePrice'].values

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_data, Y_data, test_size=0.3, random_state=100)

    # fitting and check the performance of each model
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

    tc = time.time()

    new_x_train, new_x_test = stk_train(model, X_train, X_valid, Y_train, kf, num_k_fold)

    # check the rmse of ensemble learning
    # 调参
    # stacking = LinearRegression()
    # stacking.fit(new_x_train, Y_train)
    # Y_reg_pred = stacking.predict(new_x_test)
    # r_mse = sqrt(mean_squared_error(Y_valid, Y_reg_pred))
    # print(f'stacking model\'s RMSE :{r_mse:.4f}')
    # tc = time.time() - tc
    # print('time:', tc)

    # final predict
    final_x_train, final_x_test = stk_train(model, X_train, X_final_test, Y_train, kf, num_k_fold)
    stacking = LinearRegression()
    stacking.fit(final_x_train, Y_train)
    Y_final_pred = stacking.predict(final_x_test)
