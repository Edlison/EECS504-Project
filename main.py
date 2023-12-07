# @Author  : Edlison
# @Date    : 12/3/23 02:47
from nl.regression import init_data, init_nlmodel, plot_nlmodel

if __name__ == '__main__':
    data_nl, data_gdp, x, y_true = init_data(length=20)
    print('init data')
    model = init_nlmodel(data_nl, data_gdp)
    print('init model')
    y_pred = model.predict(x)[0]
    print("Predicted GDP:", y_pred, "True GDP: ", y_true)  # After: 227.3654501276809, PRE: 133.33174064404713
    plot_nlmodel(model, data_nl, data_gdp)
    res_mse = eval(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    print("MSE: ", res_mse)  # After: 4586, PRE: 26165
