from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Gaussian Naive Bayes
    # 主要用於連續變數，比方說特徵長度為幾公分、重量為幾公斤等等。

    model = GaussianNB()
    model.fit(X_train, y_train)
    a = model.predict(X_test)

    print(a)

    b = model.predict_proba(X_test)

    print(b)

    print("Training Set Score:", model.score(X_train, y_train))
    print("Test Set Score:", model.score(X_test, y_test))


