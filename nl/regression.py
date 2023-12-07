import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from nl.data import NL_DATA, GDP_DATA


def init_data(length):
    data_nl = NL_DATA[:length]
    data_gdp = GDP_DATA[:length]
    light_array = np.array(data_nl).reshape(-1, 1)
    gdp_array = np.array(data_gdp)
    x = np.array(NL_DATA[length]).reshape(-1, 1)
    y_true = np.array(GDP_DATA[length])
    return light_array, gdp_array, x, y_true


def init_nlmodel(data_nl, data_gdp):
    light_array = np.array(data_nl).reshape(-1, 1)
    gdp_array = np.array(data_gdp)
    model = LinearRegression()
    model.fit(light_array, gdp_array)
    return model


def plot_nlmodel(model, data_nl, data_gdp):
    light_fit = np.linspace(data_nl.min(), data_nl.max(), 100).reshape(-1, 1)
    gdp_fit = model.predict(light_fit)

    plt.scatter(data_nl, data_gdp, color='blue', label='real data')
    plt.plot(light_fit, gdp_fit, color='red', label='regression')

    plt.legend()

    plt.title('Before Correlation')
    plt.xlabel('Light')
    plt.ylabel('GDP')

    plt.show()
    print('Plot!')


def eval(y_true, y_pred):
    res = mean_squared_error(y_true, y_pred)
    return res


if __name__ == '__main__':
    data_nl, data_gdp, x, y_true = init_data(length=20)
    print('init data')
    # print('nl: ', data_nl, 'gdp: ', data_gdp)
    model = init_nlmodel(data_nl, data_gdp)
    print('init model')
    y_pred = model.predict(x)[0]
    print("Predicted GDP:", y_pred, "True GDP: ", y_true)  # After: 227.3654501276809, PRE: 133.33174064404713
    plot_nlmodel(model, data_nl, data_gdp)
    res_mse = eval(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    print("MSE: ", res_mse)  # After: 4586, PRE: 26165