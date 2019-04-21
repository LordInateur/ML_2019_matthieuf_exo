# removable
import datetime

from sklearn.ensemble import RandomForestClassifier




"""
    Un classifier utilisant la methode random_forest
"""
def random_forest_classifier( points, classe, to_guess):
    start_time = datetime.datetime.now()
    # RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(points, classe)
    cl_construct_time = datetime.datetime.now() - start_time
    #  clf.predict([[.6, .6]])
    return clf.predict(to_guess), cl_construct_time.microseconds, (datetime.datetime.now() - start_time - cl_construct_time).microseconds




if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # On récupère l'ensemble de travail
    digits = datasets.load_digits()
    # On séléctionne les valeurs d'entrées (X)
    data = digits['data']
    # On selectionnes les valeurs de sortie(Y)
    target = digits['target']
    # Séparation des données d'entrainement et de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    G = [2]
    print(random_forest_classifier(x_train, y_train, x_test))


