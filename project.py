#!/usr/bin/python
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import pandas as pd
import numpy as np
import Model.stacking

if __name__ == '__main__':
    # data processing

    # model
    # initialize k-fold
    num_k_fold = 5
    kf = KFold(n_splits=num_k_fold)

    # initialize model
    model = [
        # regression

        # Linear
        # LinearRegression(),
        # rmse is too large

        # Lasso
        make_pipeline(RobustScaler(), Lasso(alpha=0.0003, random_state=100)),

        # classificaiton

        # bagging
        # Random forest
        # too slow
        RandomForestRegressor(n_estimators=3000,
                              criterion='mse',
                              max_features='sqrt',
                              min_samples_leaf=15,
                              min_samples_split=10,
                              max_depth=8),
        # boosting
        # XGBoost
        XGBRegressor(tree_method='hist',
                     max_depth=8,
                     n_estimators=3000,
                     min_child_weight=300,
                     colsample_bytree=0.8,
                     subsample=0.8,
                     eta=0.3,
                     seed=42),
        # GDT
        # too slow
        GradientBoostingRegressor(n_estimators=3000, learning_rate=0.005,
                                  max_depth=8, max_features='sqrt',
                                  min_samples_leaf=10, min_samples_split=10,
                                  loss='huber', random_state=100),
        # LightGBM
        LGBMRegressor(objective='regression', num_leaves=8,
                      learning_rate=0.005, n_estimators=3000,
                      max_bin=55, bagging_fraction=0.8,
                      bagging_freq=5, feature_fraction=0.2319,
                      feature_fraction_seed=9, bagging_seed=9,
                      min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
                      max_depth=8),
    ]

    # import data
    train_data = pd.read_csv('Data/train_finalv1_2.csv')
    train_data = train_data.drop(axis=1, columns='ID', inplace=False)

    X_data = train_data.drop(axis=1, columns='SalePrice', inplace=False).values
    Y_data = train_data['SalePrice'].values

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_data, Y_data, test_size=0.3, random_state=100)

    # fitting and check the performance of each model
    for regression in model:
        tc = time.time()
        regression.fit(X_train, Y_train)
        Y_pred = regression.predict(X_valid)
        r_mse = sqrt(mean_squared_error(Y_valid, Y_pred))
        print('model\'s RMSE :', r_mse)
        tc = time.time() - tc
        print('time:', tc)

    tc = time.time()
    new_x_train = np.zeros((X_train.shape[0], len(model)))
    new_x_test = np.zeros((X_valid.shape[0], len(model)))
    i = 0
    for regression in model:
        new_x_train[:, i], new_x_test[:, i] = Model.stacking.stacking(
            regression, X_train, Y_train, X_valid, kf, num_k_fold)
        i += 1

    # 调参
    stacking = LinearRegression()
    stacking.fit(new_x_train, Y_train)
    Y_reg_pred = stacking.predict(new_x_test)
    r_mse = sqrt(mean_squared_error(Y_valid, Y_reg_pred))
    print('stacking model\'s RMSE :', r_mse)
    tc = time.time() - tc
    print('time:', tc)
