from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model

if __name__ == '__main__':
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    model = linear_model.LogisticRegression()

    model.fit(X_train, y_train)

    a = model.predict(X_test)

    b = model.predict_proba(X_test)

    print(model.score(X_train, y_train))

    print(model.score(X_test, y_test))
