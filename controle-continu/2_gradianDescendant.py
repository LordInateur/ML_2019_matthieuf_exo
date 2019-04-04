import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import pandas
import csv
import matplotlib.pyplot as plt
import functools
from datetime import date, datetime
from decimal import *


def Hypothese(t0, t1, x):
    return t0 + t1 * x


def AllSumHyp(array, t0, t1, isT1=False):
    j = 0
    allHyp = 0
    while (j < len(array)):
        if isT1:
            allHyp += (Hypothese(t0, t1, array[j][0]) - array[j][1]) * array[j][0]
        else:
            allHyp += Hypothese(t0, t1, array[j][0]) - array[j][1]
        j += 1

    return allHyp


def DisplayGraph(array, x, y):
    m = 0

    plt.figure(1)
    plt.subplot(211)
    plt.scatter(x, y, color='red')
    plt.plot(x, reg.predict(x.reshape(-1,1)), color='blue')



    plt.subplot(212)
    plt.scatter(x, [ (y[index] - reg.predict(np.array(x_).reshape(-1,1))) for index, x_ in enumerate(x)] , color='red')
    plt.axhline(0, color='blue')

    plt.show()


t0 = 0
t1 = 0

time_stamp0 = date(2019, 3, 16).toordinal()
y_0 = 81682
time_objecitf = date(2019, 4, 5).toordinal() - time_stamp0
print("time_objecitf : ", time_objecitf)


# ['2019-03-16', 81682], ['2019-03-18', 81720], ['2019-03-20', 817160],
# Pas le temps de mettre toute les dates mais voici les 3 dernireres
array = [
        [date(2019, 3, 16).toordinal()- time_stamp0, 81682-y_0], [date(2019, 3, 18).toordinal()- time_stamp0, 81720-y_0], [date(2019, 3, 20).toordinal()- time_stamp0, 81760-y_0],
        [date(2019, 3, 28).toordinal()- time_stamp0, 81900-y_0], [date(2019, 3, 30).toordinal()- time_stamp0, 81933-y_0], [date(2019, 4, 3).toordinal()- time_stamp0, 82003-y_0]]

i = 1
lastCost = Decimal(1.0)
max_i = 7000

x = np.array([elem[0] for elem in array])
y = np.array([elem[1] for elem in array])
print(f'x:{x}')
print(f'y:{y}')

reg = LinearRegression().fit(x.reshape(-1,1),y.reshape(-1,1))
"""
while i < max_i:
    #
    print("Boucle %f" % i)
    t0inter = t0 - (1 / i) / len(array) * AllSumHyp(array, t0, t1)
    t1inter = t1 - (1 / i) / len(array) * AllSumHyp(array, t0, t1, True)
    i += 0.05

    cost = Decimal(1 / Decimal(2 * len(array)) * Decimal(AllSumHyp(array, t0, t1) ** 2))

    print("TETA0 %f" % t0inter)
    print("TETA1 %f" % t1inter)
    # print("Cost %f" % round(cost, 6))
    print("Cost %f" % cost)

    if (cost < lastCost ):
        i = max_i

    else :
        lastCost = cost
        t0 = t0inter
        t1 = t1inter

# prevision :  -1.0742676592443442e+52
print ( 'prevision : ', t0 + t1 * time_objecitf)
"""

DisplayGraph(array, x, y)