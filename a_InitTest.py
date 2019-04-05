import numpy
import scipy
import sklearn
import pandas
import csv
import matplotlib.pyplot as plt

print(" Hello word")

csv_array = []

with open('data/matchinfo.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        csv_array.append(row)
        # print(', '.join(row))


colomn = csv_array.pop(0)
print(colomn)
print(csv_array)
ar = numpy.array(csv_array)
df = pandas.DataFrame(ar, columns = colomn)


# ar = n²



