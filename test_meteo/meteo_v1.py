
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from api import getAllData

# ok si la précision est inférieur à 4
def getPrecision(y_predict, y_test):
    guested = 0
    for p in range(len(y_test)):
        guested += abs(y_test[p] - y_predict[p])
    return guested / len(y_test)


if __name__ == "__main__":
    """
        Le but est de prédire la temperature en fonction des 29 dernières valeurs
    """

    # le nombre de jour précédent dans le vecteur
    NB_VALEUR_VECTEUR = 720
    CARACTERISTIQUE_CIBLE = "tmoy" # tmin tmax

    # On récupère les données
    datas = getAllData()

    # Initialisation des variables
    data_tmax, target_tmax = [], []

    # On génère les données d'entrainements
    # TODO Mettre à jour avec la nouvelle version de l'algo (e.g.: v2)
    for i in range(len(datas)-NB_VALEUR_VECTEUR):
        data_tmax.append(list(map(lambda x : x[CARACTERISTIQUE_CIBLE], datas[i:i + NB_VALEUR_VECTEUR - 1])))
        target_tmax.append(round(datas[i+NB_VALEUR_VECTEUR][CARACTERISTIQUE_CIBLE]))

    x_train, x_test, y_train, y_test = train_test_split(data_tmax, target_tmax, test_size=0.2)

    # On se base sur un classifieur tout fait ;)
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_train, y_train)

    # On affiche l'écart moyen entre la valeur attendue et la prediction
    y_predict = clf.predict(x_test)
    print("Ecart moyen entre la prediction et la valeur attendue : ", getPrecision(y_predict, y_test))
    """ ecart moyen constaté : 1.9° à 2.2° """

