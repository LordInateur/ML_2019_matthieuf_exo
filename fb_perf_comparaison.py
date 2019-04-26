import datetime

from sklearn import datasets
from sklearn.model_selection import train_test_split

from f_forester import random_forest_classifier
from c_multiplePart import o_a_classifier
from f_svm import svm_classifier
from h_reuronal_network import neuronal_classifier_oneTraining, neuronal_classifier_stepByStep

""" Calcul la precision en %"""
def cp (y_predict, y_test):
    guested = 0
    for p in range(len(y_test)):
        if y_test[p] == y_predict[p]:
            guested += 1
    return guested * 100 / len(y_test)

if __name__ == "__main__":

    # On récupère l'ensemble de travail
    digits = datasets.load_digits()
    # On séléctionne les valeurs d'entrées (X)
    data = digits['data']
    # On selectionnes les valeurs de sortie(Y)
    target = digits['target']

    classifier = [
        ("Precision guested_tree :.......... ", random_forest_classifier, ""),
        ("Precision guested_o_a : ...........", o_a_classifier, ""),
        ("Precision guested_svm(ovo) : ......", svm_classifier, ""),
        ("Precision guested_svm(ova) : ......", svm_classifier, "ova"),
        ("Precision guested_neuronal(oneT) : ", neuronal_classifier_oneTraining, ""),
        ("Precision guested_neuronal(sbs) :  ", neuronal_classifier_stepByStep, "")
    ]
    test_data = []
    for i in range(10):
        test_data.append(train_test_split(data, target, test_size=0.2))

    for p in range(len(classifier)):
        result = (0, 0, 0)
        for i in range(10):
            if(len(classifier[p][2])>0):
                predict, compute_time, guesting_time = classifier[p][1](test_data[i][0], test_data[i][2], test_data[i][1], classifier[p][2])
            else :
                predict, compute_time, guesting_time = classifier[p][1](test_data[i][0], test_data[i][2], test_data[i][1])
            result = (result[0] + cp(predict, test_data[i][3]),
                      result[1] + compute_time,
                      result[2] +  guesting_time)

        print(classifier[p][0], result[0]/10, " (", int(result[1]/10), "+", int(result[2]/10),"microseconds)")

    """
                                                           ( fit    + predict )
    Precision guested_tree :..........  97.52777777777779  ( 209017 +  10663 microseconds)
    Precision guested_o_a : ........... 95.52777777777779  ( 322449 +    198 microseconds)
    Precision guested_svm(ovo) : ...... 98.6388888888889   (  65675 +  19589 microseconds)
    Precision guested_svm(ova) : ...... 98.6388888888889   (  65916 +  19196 microseconds)
    Precision guested_neuronal(oneT) :  79.36111111111111  ( 105299 + 166953 microseconds)
    Precision guested_neuronal(sbs) :   81.36111111111111  ( 548765 + 130554 microseconds)
    """
