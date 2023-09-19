import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model



if __name__ == '__main__':
    # 平均一天喝幾杯含糖飲料以上未來易得糖尿病
    data = pd.read_csv('LogR_data.csv')
    X = data['Amount'].values
    y = data['Result'].values

    X = X.reshape(-1, 1)


    # logistic regression
    model = linear_model.LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
            verbose=0, warm_start=False)

    model.fit(X, y)

    # intercept -> 截距
    # coef -> 係數
    print('coef', model.coef_)
    print('intercept', model.intercept_)


    w1 = float(model.coef_)
    w0 = float(model.intercept_)

    def sigmoid(x, w0, w1):
        ln_odds = w0 + w1*x
        return 1/(1+np.exp(-ln_odds))

    x = np.arange(0, 10, 1)
    s_x = sigmoid(x, w0, w1)
    plt.plot(x, s_x)
    plt.axhline(y=0.5, ls='dotted', color='k')
    # plt.show()

    a = model.predict([[0], [1], [2], [3]])
    
    new_input = np.array([[0], [1], [3], [2], [1], [1], [0], [0], [1], [0], [2], [3], [3], [2], [2], [0], [2], [3], [2], [3]])
    
    print(a)

    b = model.predict_proba(X)

    # 正反機率
    print(b)

    # 正確率
    print(model.score(X, y))
