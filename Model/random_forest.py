from sklearn.ensemble import RandomForestRegressor


class RF:
    def __init__(self):


    def rf_model(self):
        # creat model
        my_model = RandomForestRegressor()
        # training
        my_model.fit(train_X, train_y)

        test_X = test_data[predictor_cols]
        predicted_prices = my_model.predict(test_X)
        print(predicted_prices)

    # https://blog.csdn.net/Feng512275/article/details/84201993
