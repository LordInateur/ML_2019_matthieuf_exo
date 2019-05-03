import datetime

from numpy.random.mtrand import randint
from pygame import *

from sklearn import datasets
from sklearn.model_selection import train_test_split

from f_forester import random_forest_classifier
from c_multiplePart import o_a_classifier
from f_svm import svm_classifier

import numpy as np


# couleurs

black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)


""" Calcul la precision en %"""
def cp (y_predict, y_test):
    guested = 0
    for p in range(len(y_test)):
        if y_test[p] == y_predict[p]:
            guested += 1
    return guested * 100 / len(y_test)

def fn(value):
    if value > 3 :
        return True
    else :
        return False

"""
    Un classifier utilisant la methode neuronal, full training
"""
def nc_ot_poids_updateur(image, classe, liste_des_poids):
    taille_vecteur = len(image)
    # for p in range(nb_classe): # 10 : on suppose qu'il y a 10 classe qui sont des entiers
    # pour tout les points on va chercher a mettre ajour notre predicteur

    sum = 0  # le nombre de point dans le juste
    for j in range(taille_vecteur):
        point_allume = fn(image[j])
        if point_allume:
            # si le point est allumé
            sum += 1

    # Maintenant on met a jour la liste des poids
    for j in range(taille_vecteur):
        if fn(image[j]):
            # si le point est allumé
            # le but de cette fonction est de maintenir une somme du vecteur de poid identique
            liste_des_poids[classe][j] += (taille_vecteur - sum) / taille_vecteur
        else:
            liste_des_poids[classe][j] -= sum / taille_vecteur
    return liste_des_poids


def neuronal_classifier_oneTraining(points, classe, to_guess):
    start_time = datetime.datetime.now()

    taille_vecteur = len(points[0])
    nb_vecteur = len(points)
    nb_classe = 10
    liste_des_poids = [[100.]*taille_vecteur for lm in range(nb_classe)]
    # print(1, "-", liste_des_poids)
    for i in range(nb_vecteur):
        liste_des_poids = nc_ot_poids_updateur(points[i], classe[i], liste_des_poids)
    cl_construct_time = datetime.datetime.now() - start_time

    # Maintenant on essai de prédir
    ret_list = []
    for i in range(len(to_guess)) :
        max_sum = 0
        ret_class = 0
        for p in range(nb_classe):
            cur_sum = 0
            for j in range(taille_vecteur):
                cur_sum += int(fn(to_guess[i][j])) * liste_des_poids[p][j]
            if cur_sum > max_sum :
                max_sum = cur_sum
                ret_class = p
        ret_list.append(ret_class)
    return ret_list, cl_construct_time.microseconds, (datetime.datetime.now() - start_time - cl_construct_time).microseconds



"""
    Un classifier utilisant la methode neuronal, step by step
"""
def predict(poids, image):
    """"""
    max_sum = 0
    ret_class = 0
    for p in range(len(poids)):
        cur_sum = 0
        for j in range(len(image)):
            cur_sum += int(fn(image[j])) * poids[p][j]
        # print(p, "- ", cur_sum, "/", max_sum)
        if cur_sum > max_sum:
            max_sum = cur_sum
            ret_class = p
    return ret_class


def nc_sbs_poids_updateur(image, classe, liste_des_poids):

    taille_vecteur = len(image)

    # for p in range(nb_classe): # 10 : on suppose qu'il y a 10 classe qui sont des entiers
    # pour tout les points on va chercher a mettre ajour notre predicteur
    class_predict = predict(liste_des_poids, image)
    if (class_predict != classe):
        sum = 0  # le nombre de point dans le juste
        for j in range(taille_vecteur):
            point_allume = fn(image[j])
            # if p == classe[i] :
            # si le point doit être allumé
            if point_allume:
                # si le point est allumé
                sum += 1

        # Maintenant on met a jour la liste des poids pour ce notre cible
        for j in range(taille_vecteur):
            # if p == classe[i]:
            # si le point doit être allumé
            # print("cur ", "-", cur_list_des_poid)
            if fn(image[j]):
                # si le point est allumé
                # print("+(",sum, ") - ",  (taille_vecteur-sum) / taille_vecteur)
                liste_des_poids[classe][j] += (taille_vecteur - sum) / taille_vecteur
            else:
                liste_des_poids[classe][j] -= sum / taille_vecteur

        # Maintenant on met a jour la liste des poids pour celui qui a été trouvé

        for j in range(taille_vecteur):
            # if p == classe[i]:
            # si le point doit être allumé
            # print("cur ", "-", cur_list_des_poid)
            if fn(image[j]):
                # si le point est allumé
                # print("+(",sum, ") - ",  (taille_vecteur-sum) / taille_vecteur)
                liste_des_poids[class_predict][j] -= (taille_vecteur - sum) / taille_vecteur
            else:
                liste_des_poids[class_predict][j] += sum / taille_vecteur

    return liste_des_poids

def neuronal_classifier_stepByStep(points, classe, to_guess, nb_passage=2):
    start_time = datetime.datetime.now()

    taille_vecteur = len(points[0])
    nb_vecteur = len(points)
    nb_classe = 10
    liste_des_poids = [[100.]*taille_vecteur for lm in range(nb_classe)]
    # print(1, "-", liste_des_poids)
    for p in range(nb_passage):
        for i in range(nb_vecteur):
            liste_des_poids = nc_sbs_poids_updateur(points[i], classe[i], liste_des_poids)

    cl_construct_time = datetime.datetime.now() - start_time

    # Maintenant on essai de prédir
    ret_list = []
    for i in range(len(to_guess)) :
        ret_list.append(predict(liste_des_poids, to_guess[i]))
    return ret_list, cl_construct_time.microseconds, (datetime.datetime.now() - start_time - cl_construct_time).microseconds


def neuronal_classifier_stepByStep_tomax(points, classe, to_guess):
    start_time = datetime.datetime.now()

    taille_vecteur = len(points[0])
    nb_vecteur = len(points)
    nb_classe = 10
    liste_des_poids = [[100.]*taille_vecteur for lm in range(nb_classe)]
    last_perf, new_perf = 0, 0.00001
    nb_ite = 0
    # print(1, "-", liste_des_poids)
    while new_perf > last_perf :
        nb_ite += 1
        last_perf = new_perf
        for i in range(nb_vecteur):
            liste_des_poids = nc_sbs_poids_updateur(points[i], classe[i], liste_des_poids)

        # Maintenant on essai de prédir
        ret_list_learning = []
        for i in range(len(points)) :
            ret_list_learning.append(predict(liste_des_poids, points[i]))
        new_perf = cp(ret_list_learning, classe)

        cl_construct_time = datetime.datetime.now() - start_time
        if new_perf > last_perf:
            # Maintenant on essai de prédir
            ret_list = []
            for i in range(len(to_guess)):
                ret_list.append(predict(liste_des_poids, to_guess[i]))
    # print("nb_ite : " + str(nb_ite) + " - " + str(last_perf) + " - " + str(new_perf))

    return ret_list, cl_construct_time.microseconds, (datetime.datetime.now() - start_time - cl_construct_time).microseconds



def to255(value, list):
    max_poid = value
    min_poid = value

    for val in list:
        if val > max_poid:
            max_poid = val
            # print("max:", max_poid)
        if val < min_poid:
            min_poid = val

    if(min_poid == max_poid):
        return 255
    else :
        return int((value-min_poid) * 255 / (max_poid-min_poid))

"""
    Créer car ça ne marche pas.
"""
def my_waiter(ms):
    t = pygame.time.get_ticks() + ms
    while pygame.time.get_ticks() < t: pygame.event.pump()


if __name__ == "__main__":
    import pygame
    # On récupère l'ensemble de travail
    digits = datasets.load_digits()
    # On séléctionne les valeurs d'entrées (X)
    data = digits['data']
    # On selectionnes les valeurs de sortie(Y)
    target = digits['target']
    # Séparation des données d'entrainement et de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    poid = [1]*64


    """
     Example program to show using an array to back a grid on-screen.

     Sample Python/Pygame Programs
     Simpson College Computer Science
     http://programarcadegames.com/
     http://simpson.edu/computer-science/

     Explanation video: http://youtu.be/mdTeqiWyFnc
    """
    init()

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # This sets the WIDTH and HEIGHT of each grid location
    WIDTH = 15
    HEIGHT = 15

    # This sets the margin between each cell
    MARGIN = 2
    GRID_SIZE = 8

    # Initialize pygame
    pygame.init()

    # Set the HEIGHT and WIDTH of the screen
    WINDOW_SIZE = [(HEIGHT*GRID_SIZE + MARGIN*(GRID_SIZE-1) ) * 10 + MARGIN * 9, (WIDTH * GRID_SIZE+  MARGIN*(GRID_SIZE-1))]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # Set title of screen
    pygame.display.set_caption("Neural Netword")

    # Used to manage how fast the screen updates
    # Remplacer car freez de la fenetre
    clock = pygame.time.Clock()

    # -------- Main Program Loop -----------
    taille_vecteur = len(x_train[0])
    nb_vecteur = len(x_train)
    nb_classe = 10
    liste_des_poids = [[100.] * taille_vecteur for lm in range(nb_classe)]
    # liste_des_poids = [ np.asarray([randint(0, 2) for elem in range(0, taille_vecteur)]) * taille_vecteur for lm in range(nb_classe)]

    # Algo 1
    print("Aglo 1")
    for i in range(nb_vecteur):
        pygame.display.set_caption("Neural Netword : ALgo 1 - " + str(i) + " images analyse")
        liste_des_poids = nc_ot_poids_updateur(x_train[i], y_train[i], liste_des_poids)


        # affichage du tableau
        screen.fill(BLACK)
        # Draw the grid
        for p in range(nb_classe):
            image = liste_des_poids[p]
            for row in range(GRID_SIZE):
                for column in range(GRID_SIZE):
                    color = (0, to255(image[row*GRID_SIZE + column],image), to255(255-image[row*GRID_SIZE + column],image))
                    # gris = to255(image[row * GRID_SIZE + column], image)
                    # color = (gris, gris,gris)
                    # print(color)
                    pygame.draw.rect(screen,
                                     color,
                                     [((MARGIN + WIDTH) * column + MARGIN) + (HEIGHT*GRID_SIZE + MARGIN*(GRID_SIZE-1) + MARGIN ) * p ,
                                      (MARGIN + HEIGHT) * row + MARGIN,
                                      WIDTH,
                                      HEIGHT])

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        # Limit to 60 frames per second
        my_waiter(5)

    # Algo 2
    print("Aglo 2")
    liste_des_poids = [[100.] * taille_vecteur for lm in range(nb_classe)]
    for i in range(nb_vecteur):
        pygame.display.set_caption("Neural Netword : ALgo 2 - " + str(i) + " images analyse")
        liste_des_poids = nc_sbs_poids_updateur(x_train[i], y_train[i], liste_des_poids)

        # affichage du tableau
        screen.fill(BLACK)
        # Draw the grid
        for p in range(nb_classe):
            image = liste_des_poids[p]
            for row in range(GRID_SIZE):
                for column in range(GRID_SIZE):
                    color = (0, to255(image[row * GRID_SIZE + column], image),
                             to255(255 - image[row * GRID_SIZE + column], image))
                    # gris = to255(image[row * GRID_SIZE + column], image)
                    # color = (gris, gris,gris)
                    # print(color)
                    pygame.draw.rect(screen,
                                     color,
                                     [((MARGIN + WIDTH) * column + MARGIN) + (
                                                 HEIGHT * GRID_SIZE + MARGIN * (GRID_SIZE - 1) + MARGIN) * p,
                                      (MARGIN + HEIGHT) * row + MARGIN,
                                      WIDTH,
                                      HEIGHT])

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        # Limit to 60 frames per second
        my_waiter(5)


    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()


