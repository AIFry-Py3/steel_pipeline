import csv
import operator

sample = open('steel_data.csv', 'r')

csv1 = csv.reader(sample, delimiter=',')
sort = sorted(csv1, key=operator.itemgetter(1))
print(type(sort))

for sor in sort:
    print(sor)