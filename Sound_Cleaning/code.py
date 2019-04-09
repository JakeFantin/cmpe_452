#assignment 3
import csv
import math
import cmath
import more_itertools as mit

data = []
sum = [0,0]


# import data from csv
with open('sound.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, dialect = 'excel', delimiter = ',', quoting=csv.QUOTE_NONNUMERIC)
    for line in reader:
        data.append(line)

# sum up each line
for line in data:
    sum = [sum[0]+ line[0], sum[1]+ line[1]]


# Get mean by dividing sums by length of the dataset
mean = [sum[0]/len(data), sum[1]/len(data)]


# Normalize the data by subtraction the mean
for line in data:
    line = [line[0] - mean[0], line[1] - mean[1]]


# Training that network technically

#randomish weights
weights = [1,1];
deltaW = [0,0];

# learning rate
c = 0.1

# apply math to data, single iteration
for line in data:
    # get dot product
    y = mit.dotproduct(line, weights)

    # make K
    K = y * y

    # get deltaW
    deltaW[0] = c * ((line[0] * y) - (K * weights[0]));
    deltaW[1] = c * ((line[1] * y) - (K * weights[1]));

    # update weights
    weights[0] = weights[0] + deltaW[0];
    weights[1] = weights[1] + deltaW[1];

together = []

# dot product with the input data and new weights

f = open("Readme.txt", "w+")
f.write('Final Weights')
f.write('\n%f, %f\n\n' % (weights[0], weights[1]))
f.close()
for line in data:
    together.append(mit.dotproduct(line, weights))

# writing csv value
with open('output.csv', 'w', newline='\n') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for line in together:
        spamwriter.writerow([line])

