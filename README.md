# Time Series Forecasting

- ### Notice
```
bash install.sh
conda activate time-series-forecasting-env
```


- [Python 3](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Scikit-learn](http://scikit-learn.org/stable/)
- [StatsModels](https://www.statsmodels.org/)

```python
def forecast(argv_load, argv_split, argv_model, argv_save, argv_evaluate, argv_plot)
    """
    argv_load = [Input] // ['data-water.csv'] //
        - Input --> argv_load[0] - .csv file containing time series dataset
    argv_split = [Test_Size] // [0.30] // or // [95] //
        - Test_Size --> argv_split[0] - if float ([0.0, 1.0]) -> test size
                                      - if int ( > 0) -> split_point
    argv_model = [Model_Type, Window, Lags, Order]
                // ['arima', 5, 5, (1, 2, 3)] //
        - Model_Type --> argv_test[0] - type of model for training
            - AutoRegression Model -----> 'AR', 'AutoReg', 'AutoRegression', 'auto_regression'
            - Residual Forecast Error --> 'RFE', 'ResidualForecastError', 'residual_forecast_error'
            - Arima Model --------------> 'A', 'arima', 'Arima', 'ARIMA'
        - Window --> argv_test[1]
        - Lags --> argv_train[2]
        -Order --> argv_test[3] - tuple, hyper-parameter for ARIMA models
    argv_save = [Save, Des_File] // [True, 'predictions.csv'] //
        - Save --> boolean, if True, save the predictions to .csv file
    argv_evaluate = [Metrics] // ['MAE'] //
        - Metrics --> argv_evaluate[0]
            - 'MAE'  --> Mean Absolute Error
            - 'MSE'  --> Mean Squared Error
            - 'RMSE' --> Root Mean Squared Error
    argv_plot = [doPlot, Plot_Method] // ['L'] //
        - doPlot --> argv_plot[0] - boolean, if True plot the results
        - Plot_Method --> argv_plot[0]
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
```
