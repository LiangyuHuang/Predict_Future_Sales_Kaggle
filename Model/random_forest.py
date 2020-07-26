

def rf_model():

    # 把要训练的数据丢进去，进行模型训练
    rf.fit(train_X, train_y)

    '''四、用测试集预测房价'''
    test_X = test_data[predictor_cols]
    predicted_prices = my_model.predict(test_X)