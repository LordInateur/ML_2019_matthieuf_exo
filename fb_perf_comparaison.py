import datetime

from sklearn import datasets
from sklearn.model_selection import train_test_split

from f_forester import random_forest_classifier
from c_multiplePart import o_a_classifier


if __name__ == "__main__":
    # On récupère l'ensemble de travail
    digits = datasets.load_digits()
    # On séléctionne les valeurs d'entrées (X)
    data = digits['data']
    # On selectionnes les valeurs de sortie(Y)
    target = digits['target']
    # Séparation des données d'entrainement et de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)



    # debut test random forest
    start_time = datetime.datetime.now()
    guested_tree = random_forest_classifier(x_train, y_train, x_test)
    guested_tree_valid = 0

    for p in range (len(y_test)):

        if y_test[p] == guested_tree[p]:
            guested_tree_valid += 1

    print("Precision guested_tree : ", guested_tree_valid * 100 / len(y_test), " (", datetime.datetime.now() - start_time,")")

    # debut test random forest
    start_time = datetime.datetime.now()
    guested_tree = random_forest_classifier(x_train, y_train, x_test)
    guested_tree_valid = 0

    for p in range (len(y_test)):

        if y_test[p] == guested_tree[p]:
            guested_tree_valid += 1

    print("Precision guested_tree : ", guested_tree_valid * 100 / len(y_test), " (", datetime.datetime.now() - start_time,")")


    # debut test o_a
    start_time = datetime.datetime.now()
    guested_o_a = o_a_classifier(x_train, y_train, x_test)
    guested_o_a_valid = 0

    for p in range (len(y_test)):

        if y_test[p] == guested_o_a[p]:
            guested_o_a_valid += 1

    print("Precision guested_o_a : ", guested_o_a_valid*100/len(y_test), " (", datetime.datetime.now() - start_time,")")
