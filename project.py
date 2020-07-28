# from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
import lightgbm as lgb
import time
import pandas as pd

# https://github.com/1mrliu/AI_Learning_Competition/tree/master/HousePices
if __name__ == '__main__':
    print('Hello Kaggle!')
    # data processing





    # model

    # initialize model

    # regression
    # Ridge
    # cannot support
    # model_ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    # Lasso
    model_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

    # classificaiton

    # bagging
    # Random forest
    # too slow
    model_rf = RandomForestRegressor(n_estimators=3000,
                                     criterion='mse',
                                     max_features='sqrt',
                                     min_samples_leaf=15,
                                     min_samples_split=10,
                                     max_depth=8)

    # boosting
    # XGBoost
    model_xgb = xgb.XGBRegressor(tree_method='hist',
                                 max_depth=8,
                                 n_estimators=3000,
                                 min_child_weight=300,
                                 colsample_bytree=0.8,
                                 subsample=0.8,
                                 eta=0.3,
                                 seed=42)
    # GDT
    # too slow
    model_gdt = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                          max_depth=8, max_features='sqrt',
                                          min_samples_leaf=15, min_samples_split=10,
                                          loss='huber', random_state=5)

    # LightGBM
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=3000,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
                                  max_depth=8)

    # import data
    train_data = pd.read_csv('Data/train_finalv1_1.csv')
    train_data = train_data.drop(axis=1, columns='ID', inplace=False)

    X_train = train_data.drop(axis=1, columns='SalePrice', inplace=False)[:1000]
    Y_train = train_data['SalePrice'][:1000]

    # test_data = pd.read_csv('Data/test_fianlv1_1.csv')
    # test_data = test_data.drop(axis=1, columns='ID', inplace=False)

    X_valid = train_data.drop(axis=1, columns='SalePrice', inplace=False)[1000:]
    Y_valid = train_data['SalePrice'][1000:]
    # X_valid = test_data.drop(axis=1, columns='SalePrice', inplace=False)
    # Y_valid = test_data['SalePrice']

    # fitting

    # model_ridge.fit(X_train, Y_train)
    # Y_pred = model_ridge.predict(X_valid)
    # accuracy_score(Y_valid, Y_pred)
    # print ('model_ridge\'s score :', clf)

    ts = time.time()
    model_lasso.fit(X_train, Y_train)
    Y_pred = model_lasso.predict(X_valid)
    rmse = sqrt(mean_squared_error(Y_valid, Y_pred))
    print('model_lasso\'s rmse :', rmse)
    ts = time.time() - ts
    print('time:', ts)

    ts = time.time()
    model_rf.fit(X_train, Y_train)
    Y_pred = model_rf.predict(X_valid)
    rmse = sqrt(mean_squared_error(Y_valid, Y_pred))
    print('model_rf\'s rmse :', rmse)
    ts = time.time() - ts
    print('time:', ts)

    ts = time.time()
    model_xgb.fit(X_train, Y_train)
    Y_pred = model_xgb.predict(X_valid)
    rmse = sqrt(mean_squared_error(Y_valid, Y_pred))
    print('model_xgb\'s rmse :', rmse)
    ts = time.time() - ts
    print('time:', ts)

    ts = time.time()
    model_gdt.fit(X_train, Y_train)
    Y_pred = model_gdt.predict(X_valid)
    rmse = sqrt(mean_squared_error(Y_valid, Y_pred))
    print('model_gdt\'s rmse :', rmse)
    ts = time.time() - ts
    print('time:', ts)

    ts = time.time()
    model_lgb.fit(X_train, Y_train)
    Y_pred = model_lgb.predict(X_valid)
    rmse = sqrt(mean_squared_error(Y_valid, Y_pred))
    print('model_lgb\'s rmse :', rmse)
    ts = time.time() - ts
    print('time:', ts)
