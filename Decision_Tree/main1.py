from sklearn import tree
from sklearn import datasets
import pydotplus 


if __name__ == '__main__':
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    clf = tree.DecisionTreeClassifier(criterion='entropy').fit(X, y)

    clf.score(X, y)

    dot_data = tree.export_graphviz(clf, out_file = None)

    graph = pydotplus.graph_from_dot_data(dot_data)

    graph.write_pdf('iris.pdf')
