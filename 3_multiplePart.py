import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#On récupère l'ensemble de travail
digits = datasets.load_digits()
#On séléctionne les valeurs d'entrées (X)
data = digits['data']
#On selectionnes les valeurs de sortie(Y)
target  = digits['target']
#Séparation des données d'entrainement et de test
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

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

#Creation du classifier 0_vs_1
classifier = LogisticRegression(solver='lbfgs').fit(x_train, y_train)
#Tableau pour faciliter les tests: on créé une paire contenant l'image et la valeur attendue
test_values = [(x_test[index],value) for index, value in enumerate(y_test)]

for elem in test_values:
    #if elem[1] < 2 :
    result = classifier.predict([elem[0]])
    print("Resultat : ", result)
    print("Attendu :", elem[1])