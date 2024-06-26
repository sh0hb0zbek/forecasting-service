if __name__ == '__main__':
    from source import *
    #
    # # load()
    # series = load('test.csv')
    # model = load('model.pkl', load_type='M')
    # lam = load('lambda.npy', load_type='L')
    #
    # # plotting()
    # plotting(series, method='all')
    #
    # # generate_series()
    # linear_time_series = generate_series(method='L', start=0, end=100) # omitting 'method' is possible
    # quadratic_time_series = generate_series(method='Q', start=0, end=100)
    # square_root_time_series = generate_series(method='SR', start=0, end=100)
    # exponential_time_series = generate_series(method='E', start=0, end=100)
    #
    # # resampling()
    # resampled_series = resampling(series, rule='Q')
    #
    # # power_transforms()
    # reciprocal = power_transforms(series, method='R')
    # reciprocal_square_root = power_transforms(series, method='RSR')
    # log = power_transforms(series, method='L')
    # square_root = power_transforms(series, method='SR')
    # no_transform = power_transforms(series, method='No')
    #
    # # train_test_split()
    # train, test = train_test_split(series, test_size=0.50)
    # multiple_split = train_test_split(series, split_type='M', n_splits=5)
    #
    # # persistence()
    # persistence_pred = persistence(train)
    #
    # # residuals()
    # resid = residuals(test, persistence_pred)
    #
    # # evaluate_error()
    # # mae = evaluate_error(test, persistence_pred, metrics='MAE')
    # # mse = evaluate_error(test, persistence_pred, metrics='MSE')
    # # rmse = evaluate_error(test, persistence_pred, metrics='RMSE')
    #
    # # grid_search_arima()
    # best_conf, best_score = grid_search_arima(train, test, metrics='RMSE', boxcox_transformed=True,
    #                                           p_values=range(4), d_values=range(2), q_values=range(4))
    #
    # # lag_features()
    # lagged_series = lag_features(series, shifts=3)
    #
    # # windowing()
    # rolled_windowing = windowing(series, windowing_type='R', stats='MinMaxMeanStd')
    # expanded_windowing = windowing(series, windowing_type='E', stats='MinMaxMeanStd')
    #
    # # model_training()
    # ar_model_coef, ar_model_history = model_training(train, model_type='AutoRegression', window=15, lags=15)
    # rfe_model_coef, rfe_model_history = model_training(train, model_type='ResidualForecastError', window=15, lags=15)
    # _, a_model_history = model_training(train, model_type='Arima')
    #
    # # model_testing()
    # ar_pred = model_testing(model_type='AutoRegression', test=test, window=15,
    #                         model_history=ar_model_history, model_coef=ar_model_coef)
    # rfe_pred = model_testing(model_type='Residual_ForecastError', test=test, window=15,
    #                          model_history=rfe_model_history, model_coef=rfe_model_coef, shifts=2)
    # a_pred = model_testing(model_type='Arima', test=test, model_history=a_model_history,
    #                        order=(1,1,1), boxcox_transformed=True)
    #
    # # save_pred()
    # save_pred(ar_pred, 'auto_regression_predictions.csv')
    # save_pred(rfe_pred, 'residual_forecast_error_predictions.csv')
    # save_pred(a_pred, 'arima_predictions.csv')

    # forecast()
    model_arima_1 = [
        ['data-water.csv'],
        ['Arima', 'arima_train', 5, 5, (0, 1, 1), True, 0.25],
        [False],
        ['RMSE'],
        [True, 'L']
    ]
    model_arima_2 = [
        ['data-champagne.csv'],
        ['Arima', 'arima_test', 5, 5, (2, 0, 2), False],
        [False],
        ['RMSE'],
        [True, 'L']
    ]
    model_ar_1 = [
        ['data-water.csv'],
        ['AR', 'auto_regression_train', 29, 29, None, True, -7],
        [False],
        ['RMSE'],
        [True, 'L']
    ]
    model_ar_2 = [
        ['data-champagne.csv'],
        ['AR', 'auto_regression_test', 29, 29, None, False],
        [False],
        ['RMSE'],
        [True, 'L']
    ]
    model_ma_1 = [
        ['data-water.csv'],
        ['MA', 'moving_average', 5, 5, None, True, 0.25],
        [False],
        ['RMSE'],
        [True, 'L']
    ]
    model_ma_2 = [
        ['data-water.csv'],
        ['MA', 'moving_average', 5, 5, None, False],
        [False],
        ['RMSE'],
        [True, 'L']
    ]
    models = list()
    models.append(model_arima_1)
    models.append(model_arima_2)
    models.append(model_ar_1)
    models.append(model_ar_2)
    # models.append(model_ma_1)
    # models.append(model_ma_2)
    for model in models:
        forecast(model[0], model[1], model[2], model[3], model[4])
