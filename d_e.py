
import numpy as np
import matplotlib.pyplot as plt


"""
    Dans le but de trouver e
"""
if __name__ == "__main__":
    print("BRUT FORCE")

    a = 1.
    step = 1
    nb_ite = 0
    precision = 0.000000000000001


    while step > precision :

        while np.log(a) < 1 :
            a += step
            nb_ite += 1

        a-= step
        step = step/3

    print("e^x =", a)
    print("precision : ", precision )
    print(nb_ite)

    print("TAYLOR")

    polts_x = []
    polts_y = []

    def exponential(n, x):
        # initialize sum of series
        sum = 1.0
        for i in range(n, 0, -1):
            sum = 1 + x * sum / i
            polts_x.append(n-i)
            polts_y.append(sum)
            # print("sum:", sum, " n:", i)
        return sum

    def exponential_me(p, x):
        # somme de x^k / k!

        # iteration 1 pré defini car sinon si je multiplie mon fact courent par 0 je le detruit
        porduit = 1.
        current_fact = 1
        # polts_x.append(0)
        # polts_y.append(1)


        for k in range(1, p):
            current_fact *= k
            porduit += np.power(x, k) / current_fact
            # polts_x.append(k)
            # polts_y.append(sum)

        return porduit

    n = 100
    x = 1.
    print("e^x =", exponential_me(100, 1))
    print("nb_division =", n)

    """
    plt.plot(polts_x, polts_y)
    plt.show()
    """

    polts_x = []
    polts_y = []

    for p in range(1, 10):
        polts_x.append(p)
        polts_y.append(exponential_me(100, p))

    plt.plot(polts_x, polts_y)
    plt.show()



