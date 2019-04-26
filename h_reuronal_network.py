import datetime
from pygame import *

from sklearn import datasets
from sklearn.model_selection import train_test_split

from f_forester import random_forest_classifier
from c_multiplePart import o_a_classifier
from f_svm import svm_classifier


# couleurs
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)


def fn(value):
    if value > 3 :
        return True
    else :
        return False


"""
    Un classifier utilisant la methode neuronal, full training
"""
def neuronal_classifier_oneTraining(points, classe, to_guess):
    start_time = datetime.datetime.now()

    taille_vecteur = len(points[0])
    nb_vecteur = len(points)
    nb_classe = 10
    liste_des_poids = [[100.]*taille_vecteur for lm in range(nb_classe)]
    # print(1, "-", liste_des_poids)
    for i in range(nb_vecteur):
        p = int(classe[i])
        # for p in range(nb_classe): # 10 : on suppose qu'il y a 10 classe qui sont des entiers
        # pour tout les points on va chercher a mettre ajour notre predicteur

        sum = 0 # le nombre de point dans le juste
        for j in range(taille_vecteur):
            point_allume = fn(points[i][j])
            # if p == classe[i] :
            # si le point doit être allumé
            if point_allume :
                # si le point est allumé
                sum += 1

        # Maintenant on met a jour la liste des poids
        for j in range(taille_vecteur):
            # if p == classe[i]:
            # si le point doit être allumé
            # print("cur ", "-", cur_list_des_poid)
            if fn(points[i][j]):
                # si le point est allumé
                # print("+(",sum, ") - ",  (taille_vecteur-sum) / taille_vecteur)
                liste_des_poids[p][j] +=  (taille_vecteur-sum) / taille_vecteur
            else :
                liste_des_poids[p][j] -=  sum / taille_vecteur
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
            # print(p, "- ", cur_sum, "/", max_sum)
            if cur_sum > max_sum :
                max_sum = cur_sum
                ret_class = p
        ret_list.append(ret_class)
    # print(len(liste_des_poids))
    # print(len(liste_des_poids[0]))
    # print(liste_des_poids[0])
    # print(liste_des_poids)
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


def neuronal_classifier_stepByStep(points, classe, to_guess):
    start_time = datetime.datetime.now()

    taille_vecteur = len(points[0])
    nb_vecteur = len(points)
    nb_classe = 10
    liste_des_poids = [[100.]*taille_vecteur for lm in range(nb_classe)]
    # print(1, "-", liste_des_poids)
    for i in range(nb_vecteur):
        p = int(classe[i])
        # for p in range(nb_classe): # 10 : on suppose qu'il y a 10 classe qui sont des entiers
        # pour tout les points on va chercher a mettre ajour notre predicteur
        if(predict(liste_des_poids, points[i]) != p):
            sum = 0 # le nombre de point dans le juste
            for j in range(taille_vecteur):
                point_allume = fn(points[i][j])
                # if p == classe[i] :
                # si le point doit être allumé
                if point_allume :
                    # si le point est allumé
                    sum += 1

            # Maintenant on met a jour la liste des poids
            for j in range(taille_vecteur):
                # if p == classe[i]:
                # si le point doit être allumé
                # print("cur ", "-", cur_list_des_poid)
                if fn(points[i][j]):
                    # si le point est allumé
                    # print("+(",sum, ") - ",  (taille_vecteur-sum) / taille_vecteur)
                    liste_des_poids[p][j] +=  (taille_vecteur-sum) / taille_vecteur
                else :
                    liste_des_poids[p][j] -=  sum / taille_vecteur
    cl_construct_time = datetime.datetime.now() - start_time

    # Maintenant on essai de prédir
    ret_list = []
    for i in range(len(to_guess)) :
        ret_list.append(predict(liste_des_poids, to_guess[i]))
    # print(len(liste_des_poids))
    # print(len(liste_des_poids[0]))
    # print(liste_des_poids[0])
    # print(liste_des_poids)
    return ret_list, cl_construct_time.microseconds, (datetime.datetime.now() - start_time - cl_construct_time).microseconds



if __name__ == "__main__":
    # On récupère l'ensemble de travail
    digits = datasets.load_digits()
    # On séléctionne les valeurs d'entrées (X)
    data = digits['data']
    # On selectionnes les valeurs de sortie(Y)
    target = digits['target']
    # Séparation des données d'entrainement et de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    poid = [1]*64

    print(x_train[0])
    print(x_train[1])
    print(x_train[2])
    print(len(x_train[0]))

    init()
    """
    grille de 64*64 pixels
    10 px par bloque + 2 entre chaque = 640 + 128 = 768
    
    
    """

    # This sets the WIDTH and HEIGHT of each grid location
    """
     Example program to show using an array to back a grid on-screen.

     Sample Python/Pygame Programs
     Simpson College Computer Science
     http://programarcadegames.com/
     http://simpson.edu/computer-science/

     Explanation video: http://youtu.be/mdTeqiWyFnc
    """
    import pygame

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # This sets the WIDTH and HEIGHT of each grid location
    WIDTH = 50
    HEIGHT = 50

    # This sets the margin between each cell
    MARGIN = 2
    GRID_SIZE = 8

    # Create a 2 dimensional array. A two dimensional
    # array is simply a list of lists.
    """
    grid = []
    for row in range(GRID_SIZE):
        # Add an empty array that will hold each cell
        # in this row
        grid.append([])
        for column in range(GRID_SIZE):
            grid[row].append(0.)  # Append a cell

    # Set row 1, cell 5 to one. (Remember rows and
    # column numbers start at zero.)
    # grid[1][5] = 1.
    """

    # Initialize pygame
    pygame.init()

    # Set the HEIGHT and WIDTH of the screen
    WINDOW_SIZE = [HEIGHT*GRID_SIZE + MARGIN*(GRID_SIZE-1), WIDTH * GRID_SIZE+  MARGIN*(GRID_SIZE-1)]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # Set title of screen
    pygame.display.set_caption("Array Backed Grid")

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # -------- Main Program Loop -----------
    """
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # User clicks the mouse. Get the position
                pos = pygame.mouse.get_pos()
                # Change the x/y screen coordinates to grid coordinates
                column = pos[0] // (WIDTH + MARGIN)
                row = pos[1] // (HEIGHT + MARGIN)
                # Set that location to one
                grid[row][column] = 1
                print("Click ", pos, "Grid coordinates: ", row, column)

        # Set the screen background
        screen.fill(BLACK)

        # Draw the grid
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                color = WHITE
                if grid[row][column] == 1:
                    color = GREEN
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])

        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
    """



    for image in x_train :
        screen.fill(BLACK)
        # Draw the grid
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                color = (0, int(image[row*GRID_SIZE + column]*16*255/256),int((255-image[row*GRID_SIZE + column]*16)*255/256))
                # print(color)
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])

        # Limit to 60 frames per second
        clock.tick(1)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()



    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()


