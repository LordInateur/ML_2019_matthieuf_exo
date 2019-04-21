import datetime

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
#Ensemble des X qui correspondent a un 0
class0 = [x_train[index] for index, value in enumerate(y_train) if value == 0]

#Ensemble des X qui correspondent a un 1
class1 = [x_train[index] for index, value in enumerate(y_train) if value == 1]

#Creation des correspondances de sortie
values = [0]*len(class0) + [1]*len(class1)
#On rassemble les éléments d'entrée dans une seule liste
entrees = class0 + class1
"""
def o_a_classifier( points, classe, to_guess, max_iter=1000):
    start_time = datetime.datetime.now()

    #Creation du classifier 0_vs_1
    classifier = LogisticRegression(solver='lbfgs', max_iter=max_iter).fit(points, classe)
    cl_construct_time = datetime.datetime.now() - start_time
    """
    #Tableau pour faciliter les tests: on créé une paire contenant l'image et la valeur attendue
    test_values = [(x_test[index],value) for index, value in enumerate(y_test)]


    reu_site = 0

    for elem in test_values:
        #if elem[1] < 2 :
        result = classifier.predict([elem[0]])
        if result == elem[1]:
            # print("Resultat : ", result, " ( attendu : ", elem[1] , ")")
            # print(" is ok : ", result == elem[1])
            reu_site += 1
        else :
            print("Resultat : ", result, " ( attendu : ", elem[1] , ")")

    """
    return classifier.predict(to_guess), cl_construct_time.microseconds, (datetime.datetime.now() - start_time - cl_construct_time).microseconds


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
    print(o_a_classifier(x_train, y_train, x_test))



# print("Precision : ", reu_site*100/len(test_values))


