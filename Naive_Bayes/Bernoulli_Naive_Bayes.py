from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # Bernoulli Naive Bayes
    # 用在二元的特徵，比方說特徵是否出現、特徵大小、特徵長短等等這種二元的分類。

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    model = BernoulliNB()
    model.fit(X_train, y_train)
    a = model.predict(X_test)

    print(a)
    # print(y_train)

    model = BernoulliNB(binarize=1)
    model.fit(X_train, y_train)
    b = model.predict(X_test)

    print(b)
    print(model.score(X_test, y_test))

    X = pd.DataFrame(X)
    print(X)

    model = BernoulliNB(binarize=[5.8,3,4.35,1.3])
    model.fit(X_train,y_train)
    c = model.predict(X_test)

    print('Training Set Score', model.score(X_train, y_train))
    print('Test Set Score', model.score(X_test, y_test))
