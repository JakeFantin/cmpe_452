import csv
from random import random
from random import seed
from random import randrange
from math import exp


#Wine Quality Prediction - CMPE 452 Assignment 2

# Data Processing
data = []
less_clean_data = []
clean_data = []


# Opening of Data
with open('assignment2data.csv', 'rt') as csvfile:
    data = csv.reader(csvfile, dialect = 'excel', delimiter = ',')

    for wine in data:
        less_clean_data.append(wine)

    # Take out Titles
    del less_clean_data[0]

    # Change Strings to Floats
    for wine in less_clean_data:
        clean_data.append([float(i) for i in wine])

    # Change Classes to Work with
    for wine in clean_data:
        if wine[-1] == 5:
            wine[-1] = 0
        elif wine[-1] == 7:
            wine[-1] = 1
        else:
            wine[-1] = 2


# ---------------------- Methods -------------------------------------

# Make a 'network' by assembling lists of weights for each Network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Activation Calculation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# Getting the Sigmoid from the Activation value for a Node
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, wine):
    inputs = wine
    for layer in network:
        new_inputs = []
        for node in layer:
            activation = activate(node['weights'], inputs)
            node['output'] = transfer(activation)
            new_inputs.append(node['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative using the nature of the sigmoid
def derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in nodes
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for node in network[i + 1]:
                    error += (node['weights'][j] * node['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                node = layer[j]
                errors.append(expected[j] - node['output'])
        for j in range(len(layer)):
            node = layer[j]
            node['delta'] = errors[j] * derivative(node['output'])


# Update weights for each layer going
def update_weights(network, wine, lr):
    for i in range(len(network)):
        inputs = wine[:-1]
        if i != 0:
            inputs = [node['output'] for node in network[i - 1]]
        for node in network[i]:
            for j in range(len(inputs)):
                node['weights'][j] += lr * node['delta'] * inputs[j]
            node['weights'][-1] += lr * node['delta']

# Train a network for either a specified number of iterations or
def train_network(network, train, lr, n_epoch, n_outputs):
    lowest_error = 10000
    for epoch in range(n_epoch):
        sum_error = 0
        for wine in train:
            outputs = forward_propagate(network, wine)
            expected = [0 for i in range(n_outputs)]
            expected[wine[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, wine, lr)
        if (sum_error > lowest_error * 1.01):
            break
        elif(sum_error < lowest_error):
            lowest_error = sum_error
        print('>epoch=%d, lrate=%.3f, sum_error=%.3f' % (epoch, lr, sum_error))

# Find the min and max values for each column
def clean_data_minmax(clean_data):
    stats = [[min(column), max(column)] for column in zip(*clean_data)]
    return stats

# Rescale clean_data columns to the range 0-1
def normalize(clean_data, minmax):
    for wine in clean_data:
        for i in range(len(wine) - 1):
            wine[i] = (wine[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Test
def test_network(network, wine):
    predictions = forward_propagate(network, wine)
    return predictions.index(max(predictions))

#Writing Out The Test Case to a File with a Confusion Matrix

def write_to_text(network,test_set):
    f = open("TestResults.txt", "w+")
    right5 = 0
    wrong57 = 0
    wrong58 = 0
    total5 = 0
    right7 = 0
    wrong75 = 0
    wrong78 = 0
    total7 = 0
    right8 = 0
    wrong85 = 0
    wrong87 = 0
    total8 = 0
    rtotal5 = 0
    rtotal7 = 0
    rtotal8 = 0
    for wine in test_set:
        prediction = test_network(network, wine)
        if wine[-1] == 0 and prediction == 0:
            right5 += 1
            rtotal5 += 1
            total5 += 1
        elif wine[-1] == 0 and prediction == 1:
            wrong57 += 1
            rtotal5 += 1
            total7 += 1
        elif wine[-1] == 0 and prediction == 2:
            wrong58 += 1
            rtotal5 += 1
            total8 += 1
        elif wine[-1] == 1 and prediction == 1:
            right7 += 1
            rtotal7 += 1
            total7 += 1
        elif wine[-1] == 1 and prediction == 0:
            wrong75 += 1
            rtotal7 += 1
            total5 += 1
        elif wine[-1] == 1 and prediction == 2:
            wrong78 += 1
            rtotal7 += 1
            total8 += 1
        elif wine[-1] == 2 and prediction == 2:
            right8 += 1
            rtotal8 += 1
            total8 += 1
        elif wine[-1] == 2 and prediction == 0:
            wrong85 += 1
            rtotal8 += 1
            total5 += 1
        else:
            wrong87 += 1
            rtotal8 += 1
            total7 += 1

        # shift back to real class numbers
        if wine[-1] == 0:

            expected = 5
        elif wine[-1] == 1:
            expected = 7
        else:
            expected = 8
        if prediction == 0:
            prediction2 = 5
        elif prediction == 1:
            prediction2 = 7
        else:
            prediction2 = 8
        f.write('Test Results')
        f.write('\nExpected=%d, Got=%d\n\n' % (expected, prediction2))
    f.write('\tPredicted 5\tPredicted 7\tPredicted 8\nActual 5\t%d\t%d\t%d\t%d\n' % (right5, wrong57, wrong58, rtotal5))
    f.write('Actual 7\t%d\t%d\t%d\t%d\n' % (wrong75, right7, wrong78, rtotal7))
    f.write('Actual 8\t%d\t%d\t%d\t%d\n' % (wrong85, wrong87, right8, rtotal8))
    f.write('\t\t\t%d\t%d\t%d\n' % (total5, total7, total8))

    class5p = right5 / (right5 + wrong57 + wrong58)
    class5r = right5 / rtotal5
    class7p = right7 / (wrong75 + right7 + wrong78)
    class7r = right7 / rtotal7
    class8p = right8 / (wrong85 + wrong87 + right8)
    class8r = right8 / rtotal8

    print('Class 8 values')
    print(wrong85)
    print(wrong87)

    print(class8r)
    print(class8p)

    f.write('\nClass 5 Recall and Percision: ' + str(class5p) + ' and ' + str(class5r))
    f.write('\nClass 7 Recall and Percision: ' + str(class7p) + ' and ' + str(class7r))
    f.write('\nClass 8 Recall and Percision: ' + str(class8p) + ' and ' + str(class8r))
    f.close()



# -------------------------- Post Method Definition Code -----------------

#Nomralizing the Data by rescaling each column from 0-1
minmax = clean_data_minmax(clean_data)
normalize(clean_data, minmax)

#Creating Training and Testing Partitions
train_set = []
test_set = []
train_set_size = len(clean_data)*0.7
for i in range(len(clean_data)):
    while len(clean_data) > train_set_size:
        index = randrange(len(clean_data))
        test_set.append(clean_data.pop(index))
train_set = clean_data

# Test training backprop algorithm
seed(1)
n_inputs = len(train_set[0]) - 1
n_outputs = len(set([wine[-1] for wine in train_set]))
network = initialize_network(n_inputs, 7, n_outputs)
train_network(network, train_set, 1, 700, n_outputs)
for layer in network:
    print(layer)

right = 0
wrong = 0
total = 0

# ------------------Test with Test set------------------------------
for wine in test_set:
    prediction = test_network(network, wine)
    if wine[-1] == prediction:
        right += 1
    else:
        wrong += 1
    total += 1
    print('Expected=%d, Got=%d' % (wine[-1], prediction))
error = wrong/total*100
print('Error: %d' % error)

# write_to_text(network, test_set)

