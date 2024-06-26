def load(filename, load_type='D', header=0, index_col=0,
         parse_dates=True, squeeze=True, date_parser=None):
    """
        'D'---> load dataset which is .csv file format
        'M'---> load model which is .pkl file format
        'N'---> load numpy value which is .npy format

        returns time series, model or lambda value respectively
    """
    if load_type in ['dataset', 'D']:
        from pandas import read_csv
        return read_csv(filename, header=header, index_col=index_col, parse_dates=parse_dates,
                        squeeze=squeeze, date_parser=date_parser)
    elif load_type in ['model', 'M']:
        from statsmodels.tsa.arima.model import ARIMAResults
        return ARIMAResults.load(filename)
    elif load_type in ['npy', 'N']:
        from numpy import load
        return load(filename)


def plotting(data, method, freq='A', interpolation=None, aspect='auto',
             start_row=None, end_row=None, title=None, lags=50, ax=None, line='r'):
    """
        'L'------> Line Plot
        'D.'-----> Dot Plot
        'H'------> Histogram
        'De'-----> Density Plot
        'S'------> Scatter Plot
        'A'------> Autocorrelation Plot
        'ACF'----> Autocorrelation Function Plot
        'PACF'---> Partial Autocorrelation Function Plot
        'QQ'-----> Q-Q Plot
        'SL'-----> Stacked Line Plot
        'B'------> Box Plot
        'HM'-----> Heat Map
    """
    from matplotlib import pyplot
    from pandas import Grouper
    from pandas import DataFrame
    if method in ['L', 'Line']:
        if isinstance(data, list):
            for i in range(len(data)):
                plt = DataFrame(data[i])
                pyplot.plot(plt)
        else:
            data.plot(ax=ax)
    if method in ['D.', 'Dot']:
        data.plot(style='k.')
    if method in ['H', 'Histogram']:
        data.hist(ax=ax)
    if method in ['De', 'Density']:
        data.plot(kind='kde', ax=ax)
    if method in ['S', 'Scatter']:
        from pandas.plotting import lag_plot
        lag_plot(data, ax=ax)
    if method in ['A', 'Autocorrelation']:
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(data, ax=ax)
    if method in ['ACF']:
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(data, lags=lags, ax=ax)
    if method in ['PACF']:
        from statsmodels.graphics.tsaplots import plot_pacf
        plot_pacf(data, lags=lags, ax=ax)
    if method in ['QQ']:
        from statsmodels.graphics.gofplots import qqplot
        qqplot(data, line=line, ax=ax)
    if method in ['SL', 'StackedLine',
                  'B', 'Box',
                  'HM', 'HeatMap']:
        groups = data[start_row:end_row].groupby(Grouper(freq=freq))
        years = DataFrame()
        for name, group in groups:
            years[name.year] = group.values
        if method in ['SL', 'StackedLine']:
            years.plot(subplots=True, legend=False)
        if method in ['B', 'Box']:
            years.boxplot()
        if method in ['HM', 'HeatMap']:
            years = years.T
            pyplot.matshow(years, interpolation=interpolation, aspect=aspect)
    if title is not None:
        pyplot.title(title)
    pyplot.show()


def generate_series(method='L', start=0, end=10):
    """
        'L'  --> Linear Time Series
        'Q'  --> Quadratic Time Series
        'SR' --> Square Root Time Series
        'E'  --> Exponential Time Series
    """
    if method in ['L', 'Linear']:
        series = [i for i in range(start, end)]
    elif method in ['Q', 'Quadratic']:
        series = [i ** 2 for i in range(start, end)]
    elif method in ['SR', 'SquareRoot']:
        from math import sqrt
        series = [sqrt(i) for i in range(start, end)]
    elif method in ['E', 'Exponential']:
        from math import exp
        series = [exp(i) for i in range(start, end)]
    else:
        series = []
    return series


def resampling(data, rule, method=None, order=None):
    """
    data: time series, pandas.Series
    rule: DateOffset, Timedelta or str
        The offset string or object representing target conversion
    method: str, default ‘linear’
        Interpolation technique to use
        it can be one of the following methods:
            'linear' 'time' 'quadratic' 'cubic'
            'slinear' 'zero' 'nearest' 'spline'
    """
    sampled = data.resample(rule=rule).mean()
    if method is not None:
        if order is not None:
            sampled = sampled.interpolate(method=method, order=order)
        else:
            sampled = sampled.interpolate(method=method)
    return sampled


def power_transforms(data, method=None):
    """
        'R'   --> Reciprocal Transform
        'RSR' --> Reciprocal Square Root Transform
        'L'   --> Log Transform
        'SR'  --> Square Root Transform
        'No'  --> No Transform
    """
    from pandas import DataFrame
    from scipy.stats import boxcox
    lmbda = {
        'R': -1.0,
        'Reciprocal': -1.0,
        'RSR': -0.5,
        'ReciprocalSquareRoot': -0.5,
        'L': 0.0,
        'Log': 0.0,
        'SR': 0.5,
        'SquareRoot': 0.5,
        'No': 1.0
    }
    dataframe = DataFrame(boxcox(data, lmbda=lmbda[method]))
    return dataframe


def train_test_split(data, split_type=None, test_size=0.33, split_point=None, n_splits=2):
    """
    Split time series into train and test sets
    split_type: there are 2 types of train&test splitting
        - simple splitting, split_type=None
        - multiple splitting, split_type='multiple' or 'multi' or 'M'
    test_size: [0.0, 1.0]
    """
    X = data.values
    if split_type is None:
        if split_point is None:
            split_point = int(len(X) * (1 - test_size))
        train, test = X[:split_point], X[split_point:]
        train = train.astype('float32')
        test = test.astype('float32')
        return train, test
    elif split_type in ['multiple', 'multi', 'M']:
        from sklearn.model_selection import TimeSeriesSplit
        splits = TimeSeriesSplit(n_splits=n_splits)
        lst = []
        for train_idx, test_idx in splits.split(X):
            lst.append((X[train_idx], X[test_idx]))
        return lst


def persistence(x):
    return [i for i in x]


def residuals(test, pred):
    from pandas import DataFrame
    residuals = [test[i] - pred[i] for i in range(len(pred))]
    return DataFrame(residuals)


def evaluate_error(test, pred, metrics='RMSE'):
    """
        'MAE'  --> Mean Absolute Error
        'MSE'  --> Mean Squared Error
        'RMSE' --> Root Mean Squared Error
    """
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    result = dict()
    if metrics in ['all', 'MeanAbsoluteError', 'MAE']:
        result['MAE'] = mean_absolute_error(test, pred)
        if metrics != 'all':
            return result['MAE']
    if metrics in ['all', 'MeanSquaredError', 'MSE']:
        result['MSE'] = mean_squared_error(test, pred)
        if metrics != 'all':
            return result['MSE']
    if metrics in ['all', 'RootMeanSquaredError', 'RMSE']:
        from math import sqrt
        if 'MSE' in result.keys():
            result['RMSE'] = sqrt(result['MSE'])
        else:
            result['RMSE'] = sqrt(mean_squared_error(test, pred))
        if metrics != 'all':
            return result['RMSE']
    if metrics == 'all':
        return result


def grid_search_arima(train, test, metrics='RMSE', boxcox_transformed=False,
                      p_values=range(10), d_values=range(3), q_values=range(3)):
    best_score, best_conf = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    _, model_history = model_training(train=train, model_type='arima')
                    pred = model_predict(model_type='arima', test=test, model_history=model_history,
                                         order=order, boxcox_transformed=boxcox_transformed)
                    rmse = evaluate_error(metrics=metrics, test=test, pred=pred)
                    if rmse < best_score:
                        best_score, best_conf = rmse, order
                except:
                    continue
    return best_conf, best_score


def lag_features(data, shifts=2, axis=1, columns=None):
    from pandas import DataFrame
    from pandas import Series
    from pandas import concat
    data = Series(data)
    temps = DataFrame(data.values)
    shifted = []
    for t in range(shifts, 0, -1):
        shifted.append(temps.shift(t - 1))
    dataframe = concat(shifted, axis=axis)
    dataframe = dataframe[shifts-1:]
    if columns is False:
        return dataframe
    if columns is None:
        columns = []
        if shifts > 2:
            for i in range(shifts-2, 0, -1):
                columns.append(f't-{i}')
        columns.append('t')
        columns.append('t+1')
    dataframe.columns = columns
    return dataframe.values


def windowing(data, windowing_type, stats='Mean', widths=2, axis=1):
    """
    data: type: pandas.Series carrying time series data
    type: types of windowing
        'R' or 'Rolling'
        'E' or 'Expanding'
    """
    from pandas import DataFrame
    from pandas import concat
    temps = DataFrame(data.values)
    if windowing_type in ['R', 'Rolling']:
        window = temps.shift(widths - 1).rolling(window=widths)
    elif windowing_type in ['E', 'Expanding']:
        window = temps.expanding()
    windowed = []
    columns = []
    if 'Min' in stats:
        windowed.append(window.min())
        columns.append('min')
    if 'Mean' in stats:
        windowed.append(window.mean())
        columns.append('mean')
    if 'Max' in stats:
        windowed.append(window.max())
        columns.append('max')
    if 'Std' in stats:
        windowed.append(window.std())
        columns.append('std')
    windowed.append(temps.shift(-1))
    columns.append('t+1')
    dataframe = concat(windowed, axis=axis)
    dataframe.columns = columns
    return dataframe


def model_training(train, model_type=None, model_name=None, lags=5, order=None):
    """
    train: carrying training time series dataset
    model_type: type of forecasting models
        AutoRegression Model -----> 'AR', 'AutoReg', 'AutoRegression', 'auto_regression'
        Residual Forecast Error --> 'RFE', 'ResidualForecastError', 'residual_forecast_error'
        Arima Model --------------> 'A', 'arima', 'Arima', 'ARIMA'
    model_name: name of forecasting model
    return:
        - AutoRegression Model:
            - model_fit, train
        - MovingAverage Model:
            - model_fit, train_resid
        - ARIMA model:
            - model_fit, lam
    """
    import numpy
    lam = None
    if model_type in ['AR', 'AutoReg', 'AutoRegression', 'auto_regression']:
        from statsmodels.tsa.ar_model import AutoReg
        model = AutoReg(train, lags=lags)
        model_fit = model.fit()
    elif model_type in ['MA', 'MovingAverage', 'moving_average']:
        train = lag_features(train, shifts=2)
        train_X = train[:, 0]
        train_Y = train[:, 1]
        train_pred = persistence(train_X)
        train_resid = residuals(train_Y, train_pred)
        model_fit, train, _ = model_training(train=train_resid, model_type='AR', model_name=model_name, lags=lags)
    elif model_type in ['A', 'arima', 'Arima', 'ARIMA']:
        from statsmodels.tsa.arima.model import ARIMA
        from scipy.stats import boxcox
        history = persistence(train)
        transformed, lam = boxcox(history)
        if lam < -5:
            transformed, lam = history, 1
        model = ARIMA(transformed, order=order)
        model_fit = model.fit()
    return model_fit, train, lam


def model_save(model_fit, train, lam, model_name):
    save(save_type='model', source=model_fit, dfile=model_name + '.pkl')
    save(save_type='npy', source=train, dfile=model_name + '.npy')
    if lam is not None:
        save(save_type='npy', source=lam, dfile=model_name + '_lam.npy')


def model_testing(model_type, model_name, test=None, window=None, order=None, lags=5):
    """
    test: carrying testing time series dataset
    model_type: type of forecasting models
        AutoRegression Model -----> 'AR', 'AutoReg', 'AutoRegression', 'auto_regression'
        Residual Forecast Error --> 'RFE', 'ResidualForecastError', 'residual_forecast_error'
        Arima Model --------------> 'A', 'arima', 'Arima', 'ARIMA'
    return: predictions
    """
    predictions = list()
    model_fit = load(model_name + '.pkl', load_type='model')
    train = load(model_name + '.npy', load_type='npy')
    if model_type in ['AR', 'AutoReg', 'AutoRegression', 'auto_regression']:
        coef = model_fit.params
        history = train[len(train) - window:]
        history = [history[i] for i in range(len(history))]
        for t in range(len(test)):
            length = len(history)
            lag = [history[i] for i in range(length-window, length)]
            yhat = coef[0]
            for d in range(window):
                yhat += coef[d + 1] * lag[window - d - 1]
            predictions.append(yhat)
            history.append(test[t])
    elif model_type in ['MA', 'MovingAverage', 'moving_average']:
        coef = model_fit.params
        history = train[len(train) - window:]
        history = [history[i] for i in range(len(history))]
        test = lag_features(test, shifts=2, columns=False)
        test_X, test_Y = test[:, :-1], test[:, -1]
        for t in range(len(test_Y)):
            yhat = test_X[t]
            error = test_Y[t] - yhat
            length = len(history)
            lag = [history[i] for i in range(length - window, length)]
            pred_error = coef[0]
            for d in range(window):
                pred_error += coef[d + 1] * lag[window - d - 1]
            yhat += pred_error
            predictions.append(yhat)
            model_history.append(error)
    elif model_type in ['A', 'arima', 'Arima', 'ARIMA']:
        from statsmodels.tsa.arima.model import ARIMA
        from scipy.special import inv_boxcox
        history = persistence(train)
        lam = load(model_name+'_lam.npy', load_type='npy')
        yhat = model_fit.forecast()[0]
        yhat = inv_boxcox(yhat, lam)
        predictions.append(yhat)
        history.append(test[0])
        for t in range(1, len(test)):
            model_fit, _, lam = model_training(history, model_type='arima', model_name=model_name, order=order)
            yhat = model_fit.forecast()[0]
            yhat = inv_boxcox(yhat, lam)
            predictions.append(yhat)
            history.append(test[t])
    return predictions


def save(save_type, source, dfile):
    """
    save_type:
        - 'pred', or 'predictions', or 'P'
        - 'model', or 'M'
        - 'numpy', or 'npy', or 'N'
    source: can be different regarding to save_type
        - for 'pred' - DataFrame containing model predictions
        - for 'model' - model_fit
        - for 'npy' - an array containing data
    dfile: destination file
        - .csv filename - for 'pred'
        - .pkl filename - for 'model'
        - .npy filename - for 'numpy'
    """
    if save_type in ['pred', 'predictions', 'P']:
        from pandas import DataFrame
        dataframe = DataFrame({'predictions': data})
        dataframe.to_csv(dfile)
    elif save_type in ['model', 'M']:
        source.save(dfile)
    elif save_type in ['numpy', 'npy', 'N']:
        import numpy
        numpy.save(dfile, source)


def forecast(argv_load, argv_model, argv_save, argv_evaluate, argv_plot):
    """
    argv_load = [Input] // ['data-water.csv'] //
        - Input --> argv_load[0] /str/ - .csv file containing time series dataset
    argv_model = if Trainind=True  -> [Model_Type, Model_Name, Window, Lags, Order, Training, Test_Size]
                 if Trainind=False -> [Model_Type, Model_Name, Window, Lags, Order, Training]
                // ['arima', 'arima', True, 0.25, 5, 5, (1, 2, 3)] /
                // ['arima', 'arima', False, 5, 5, (1, 2, 3)]     //
        - Model_Type --> argv_model[0] /str/ - type of model
            - AutoRegression Model -----> 'AR', 'AutoReg', 'AutoRegression', 'auto_regression'
            - Residual Forecast Error --> 'RFE', 'ResidualForecastError', 'residual_forecast_error'
            - Arima Model --------------> 'A', 'arima', 'Arima', 'ARIMA'
        - Model_Name --> argv_model[1] /str/ - name of model
        - Window --> argv_mode[2] - /int/
        - Lags --> argv_mode[3] - /int/
        - Order --> argv_mode[4] - /tuple/ hyper-parameter for ARIMA models
        - Training --> argv_model[5] /boolean/ - if True, it will train the model and save it
                                               - if False, it will load the model
        - Test_Size --> argv_model[6] /float or int/ - if Training is True, Test_Size is required,
                                                      - otherwise, it will be considered as None
            - /float/ [0.0, 1.0] - portion of testing set
            - /int/ - split_point
    argv_save = [Save, Des_File] // [True, 'predictions.csv'] //
        - Save --> argv_save[0] - /boolean/ if True, save the predictions to .csv file
        - Des_File --> argv_save[1] - /str/ - if Save is True, Des_File is required
                                            - otherwise it is not needed
    argv_evaluate = [Metrics] // ['MAE'] //
        - Metrics --> argv_evaluate[0] - /str/
            - 'MAE'  --> Mean Absolute Error
            - 'MSE'  --> Mean Squared Error
            - 'RMSE' --> Root Mean Squared Error
    argv_plot = [doPlot, Plot_Method] // ['L'] //
        - doPlot --> argv_plot[0] - /boolean/ if True plot the results
        - Plot_Method --> argv_plot[0] - /str/ - if doPlot is True, Plot_Method is required
                                               - otherwise it is not needed
            - 'L'------> Line Plot
            - 'D.'-----> Dot Plot
            - 'H'------> Histogram
            - 'De'-----> Density Plot
            - 'S'------> Scatter Plot
            - 'A'------> Autocorrelation Plot
            - 'ACF'----> Autocorrelation Function Plot
            - 'PACF'---> Partial Autocorrelation Function Plot
            - 'QQ'-----> Q-Q Plot
            - 'SL'-----> Stacked Line Plot
            - 'B'------> Box Plot
            - 'HM'-----> Heat Map
    """

    # import libraries
    from time import time

    start_time = time()

    # load dataset
    test = load(argv_load[0])

    if argv_model[5]:
        # split into training and testing sets
        if isinstance(argv_model[6], float):
            train, test = train_test_split(test, test_size=argv_model[6])
        else:
            train, test = train_test_split(test, split_point=argv_model[6])

        model_fit, train, lam = model_training(train=train, model_type=argv_model[0], model_name=argv_model[1],
                                            lags=argv_model[3], order=argv_model[4])
        model_save(model_fit=model_fit, train=train, lam=lam, model_name=argv_model[1])

    # predict
    pred = model_testing(model_type=argv_model[0], model_name=argv_model[1], test=test,
                         window=argv_model[2], lags=argv_model[2], order=argv_model[4])

    # save the predictions
    if argv_save[0]:
        save(save_type='predictions', source=pred, dfile=argv_save[1])

    # evaluate performance
    rmse = evaluate_error(test, pred, metrics=argv_evaluate[0])
    print('{"%s": %.3f, "Execution_time": %.3f}' % (argv_evaluate[0], rmse, time()-start_time))

    # plot results
    if argv_plot[0]:
        from source import plotting
        plotting([test, pred], method=argv_plot[1])
