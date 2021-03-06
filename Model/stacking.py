import numpy as np


def stk_train(model, X_train, X_test, Y_train, kf, num_k_fold):
    new_x_train = np.zeros((X_train.shape[0], len(model)))
    new_x_test = np.zeros((X_test.shape[0], len(model)))
    i = 0
    for regression in model:
        new_x_train[:, i], new_x_test[:, i] = stacking(
            regression, X_train, Y_train, X_test, kf, num_k_fold)
        i += 1
    return new_x_train, new_x_test


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
