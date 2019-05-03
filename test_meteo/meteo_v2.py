import datetime

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from api import getAllData

# ok si la précision est inférieur à 4
from api import dayInYearFromString


def getPrecision(y_predict, y_test):
    guested = 0
    for p in range(len(y_test)):
        guested += abs(y_test[p] - y_predict[p])
    return guested / len(y_test)

def print_temps(temp):
    """"""

if __name__ == "__main__":
    """
        Le but est de prédire la temperature en fonction des 29 dernières valeurs
    """

    print("\nDébut du script de prévision v2 ")

    # le nombre de donnée +1 de de donnée de le vecteur d'entrée
    NB_VALEUR_VECTEUR = 700
    # le mois est très important pour la température donc je le spam pour qu'il soit bien vue
    NB_REPETITION_MOIS = 700
    # le grain du groupage par jour
    NB_JOUR_MEME_NUMERO = 4
    # Les trois prédiction
    CARACTERISTIQUE_CIBLE = ["tmin", "tmoy", "tmax"]

    # Variable pour gérer l'affichage
    NOMBRE_DE_JOUR_A_CALCULER = 700
    NB_DE_JOUR_CALCULE_PAR_AFFICHE = 10

    # On récupère les données
    datas = getAllData()

    """ # test de sauvegarde dans un fichier mais l'objet ne se recharge pas bien :/
    print(datas[0])
    print(datas)
    # numpy.array(datas).sort(key=lambda x: x["date"], reverse=False)
    datas = numpy.array(datas,
                dtype=numpy.dtype([('date', str), ('dayInYear', str), ('tmin', float), ('tmoy', float), ('tmax', float)])
                )
    datas.argsort("date")
    print(datas)
    """
    print("\nDébut entrainement du classifieur ")
    # On génère les données d'entrainements


    # On se base sur un classifieur tout fait ;)
    car_cible = "tmax"
    clfs = {}
    def fn (x:dict, car_cible_):
        return x[car_cible_]
    for car_cible in CARACTERISTIQUE_CIBLE :
        clfs[car_cible] = RandomForestClassifier(n_estimators=100).fit(
            # [[int(str.split(datas[i]["date"], '-')[1]) for xxx in range(NB_REPETITION_MOIS)]+[ fn(sub, car_cible) for sub in datas[i:i + NB_VALEUR_VECTEUR - 1]] for i in range(len(datas)-NB_VALEUR_VECTEUR)],
            [[round(int(datas[i]["dayInYear"])/NB_JOUR_MEME_NUMERO) for xxx in range(NB_REPETITION_MOIS)]+[ fn(sub, car_cible) for sub in datas[i:i + NB_VALEUR_VECTEUR - 1]] for i in range(len(datas)-NB_VALEUR_VECTEUR)],
            [round(datas[i + NB_VALEUR_VECTEUR][car_cible]) for i in range(len(datas) - NB_VALEUR_VECTEUR)])


    print("Première date du jeu de donnée : " + datas[0]["date"])
    print("Dernière date du jeu de donnée : " + datas[len(datas)-1]["date"])

    for p in range(NOMBRE_DE_JOUR_A_CALCULER):
        last_data = datas[len(datas)-1]
        # print("last_data:", last_data)
        new_date = str((datetime.datetime.strptime(last_data["date"], "%Y-%m-%d") + datetime.timedelta(days=1)).date())
        dernier_index = len(datas)-NB_VALEUR_VECTEUR
        datas += [{
            'date' : new_date,
            'dayInYear' : dayInYearFromString(new_date),
            'tmin' : clfs["tmin"].predict([[round(int(datas[i]["dayInYear"])/NB_JOUR_MEME_NUMERO) for xxx in range(NB_REPETITION_MOIS)]+[ fn(sub, "tmin") for sub in datas[i:i + NB_VALEUR_VECTEUR - 1]] for i in [dernier_index]])[0],
            'tmoy' : clfs["tmoy"].predict([[round(int(datas[i]["dayInYear"])/NB_JOUR_MEME_NUMERO) for xxx in range(NB_REPETITION_MOIS)]+[ fn(sub, "tmoy") for sub in datas[i:i + NB_VALEUR_VECTEUR - 1]] for i in [dernier_index]])[0],
            'tmax' : clfs["tmax"].predict([[round(int(datas[i]["dayInYear"])/NB_JOUR_MEME_NUMERO) for xxx in range(NB_REPETITION_MOIS)]+[ fn(sub, "tmax") for sub in datas[i:i + NB_VALEUR_VECTEUR - 1]] for i in [dernier_index]])[0]
        }]

        if(dernier_index % NB_DE_JOUR_CALCULE_PAR_AFFICHE == 0):
            print(datas[len(datas)-1])




