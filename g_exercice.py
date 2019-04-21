# MNIST random forest & SVm (+rbf)
import datetime

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from c_multiplePart import o_a_classifier
from f_forester import random_forest_classifier
from f_svm import svm_classifier


""" Calcul la precision en %"""
def cp (y_predict, y_test):
    guested = 0
    for p in range(len(y_test)):
        if y_test[p] == y_predict[p]:
            guested += 1
    return guested * 100 / len(y_test)

"""
    On test les perfs de prediction sur des vecteurs
"""
if __name__ == "__main__":
    dataset = fetch_olivetti_faces()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

    # debut test random forest
    predict, compute_time, guesting_time = random_forest_classifier(x_train, y_train, x_test)
    print("Precision guested_tree : ", cp(predict, y_test), " (", compute_time, "+", guesting_time, "microseconds)")

    # debut test random forest
    predict, compute_time, guesting_time = random_forest_classifier(x_train, y_train, x_test)
    print("Precision guested_tree : ", cp(predict, y_test), " (", compute_time, "+", guesting_time, "microseconds)")

    # debut test o_a
    predict, compute_time, guesting_time = o_a_classifier(x_train, y_train, x_test)
    print("Precision guested_o_a : ", cp(predict, y_test), " (", compute_time, "+", guesting_time, "microseconds)")

    # debut test svm (ovo)
    predict, compute_time, guesting_time = svm_classifier(x_train, y_train, x_test)
    print("Precision guested_svm(ovo) : ", cp(predict, y_test), " (", compute_time, "+", guesting_time, "microseconds)")

    # debut test svm (ova)
    predict, compute_time, guesting_time = svm_classifier(x_train, y_train, x_test, 'ova')
    print("Precision guested_svm(ova) : ", cp(predict, y_test), " (", compute_time, "+", guesting_time, "microseconds)")


