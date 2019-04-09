import random
import tensorflow as tf
import numpy as np

train_txt = open('train.txt', 'r')
test_txt = open('test.txt', 'r')

# Input Holders
flowers_train = []
flowers_test = []

# Learning Rate and Weight arrays used throughout program
c = 0.5
weights1 = [0, 0, 0, 0, 0]
weights2 = [0, 0, 0, 0, 0]
weights3 = [0, 0, 0, 0, 0]

# Saved Starting Weights
sweights1 = [0, 0, 0, 0, 0]
sweights2 = [0, 0, 0, 0, 0]
sweights3 = [0, 0, 0, 0, 0]
# Best Weights to be saved during pocketing
best_weights = [[], [], [], 0]
# Recorded Classified Positives and True Positives from testing for each Flower type
setosa_ret = 0
setosa_tp = 0
versicolor_ret = 0
versicolor_tp = 0
virginica_ret = 0
virginica_tp = 0

# Iterations
iterations = 3000


# Parsing text files, replacing name with one hot encoded list
training = (train_txt.read().split('\n'))
train_txt.close()
for flower_data in training:
    flower = flower_data.split(',')
    flower[0] = float(flower[0])
    flower[1] = float(flower[1])
    flower[2] = float(flower[2])
    flower[3] = float(flower[3])
    if flower[4] == 'Iris-setosa':
        flower[4] = [1, 0, 0]
    elif flower[4] == 'Iris-versicolor':
        flower[4] = [0, 1, 0]
    else:
        flower[4] = [0, 0, 1]
    flowers_train.append(flower)

testing = (test_txt.read().split('\n'))
test_txt.close()
for flower_data in testing:
    flower = flower_data.split(',')
    flower[0] = float(flower[0])
    flower[1] = float(flower[1])
    flower[2] = float(flower[2])
    flower[3] = float(flower[3])
    if flower[4] == 'Iris-setosa':
        flower[4] = [1, 0, 0]
    elif flower[4] == 'Iris-versicolor':
        flower[4] = [0, 1, 0]
    else:
        flower[4] = [0, 0, 1]
    flowers_test.append(flower)
# ------------------------------------------------------------------------------------------------------------------
# Then I write all my methods and at the bottom I call them

# Setting weights to Random ints between 0 and 2 for all three sets, saving them to be displayed as starting weights
def set_weights():
    global weights1
    global weights2
    global weights3
    global sweights1
    global sweights2
    global sweights3
    i = 0
    while i < 5:
        # sweights = starting weights
        weights1[i] = random.randint(0, 2)
        sweights1[i] = weights1[i]
        weights2[i] = random.randint(0, 2)
        sweights2[i] = weights2[i]
        weights3[i] = random.randint(0, 2)
        sweights3[i] = weights3[i]
        i += 1


# Training model
def train_model():
    global flowers_train
    global iterations
    global weights1
    global weights2
    global weights3
    global best_weights

    streak = 0
    best_streak = 0

    i = 0

    # Checks the output (result) against expected list (flower[4]) and adjusts weights accordingly
    while i < iterations:
        true = 0
        # all_true is set to zero whenever the model makes a mistake
        all_true = 1
        for flower in flowers_train:
            # each node is a method, and uses corresponding weights
            result = [node1(flower), node2(flower), node3(flower)]
            true_flag1 = 0
            true_flag2 = 0
            true_flag3 = 0
            if result[0] > flower[4][0]:
                decrease1(flower)
                all_true = 0
            elif result[0] < flower[4][0]:
                increase1(flower)
                all_true = 0
            else:
                true_flag1 = 1
            if result[1] > flower[4][1]:
                decrease2(flower)
                all_true = 0
            elif result[1] < flower[4][1]:
                increase2(flower)
                all_true = 0
            else:
                true_flag2 = 1
            if result[2] > flower[4][2]:
                decrease3(flower)
                all_true = 0
            elif result[2] < flower[4][2]:
                increase3(flower)
                all_true = 0
            else:
                true_flag3 = 1
            if true_flag1 == 1 and true_flag2 == 1 and true_flag3 == 1:
                true += 1
                streak += 1
            else:
                # Pocketing used here
                if streak >= best_streak:
                    correct = 0
                    # checks current weights against all data and compares number correct to old best
                    for flower in flowers_train:
                        result = [node1(flower), node2(flower), node3(flower)]
                        if result == flower[4]:
                            correct += 1
                    if correct > best_weights[3]:
                        best_weights[0] = weights1
                        best_weights[1] = weights2
                        best_weights[2] = weights3
                        best_weights[3] = correct
                        best_streak = streak
                streak = 0
        # if not a single flower has a wrong classification, it breaks out of the while loop here
        if all_true == 1:
            break
        i += 1
    # checks after last iteration if the current weights are better than the saved weights
    correct_ae = 0
    for flower in flowers_train:
        result = [node1(flower), node2(flower), node3(flower)]
        if result == flower[4]:
            correct_ae += 1
    # if the current weights are not better than the best weights, switch out
    if correct_ae < best_weights[3]:
        weights1 = best_weights[0]
        weights2 = best_weights[1]
        weights3 = best_weights[2]

    # test using testing data set
    test_model()


# test using test data set
def test_model():
    global flowers_test
    global setosa_ret
    global setosa_tp
    global versicolor_ret
    global versicolor_tp
    global virginica_ret
    global virginica_tp

    correct = 0
    incorrect = 0

    for flower3 in flowers_test:
        result = [node1(flower3), node2(flower3), node3(flower3)]
        # uses output tester to take care of extra numbers
        if test_output(result, flower3[4]):
            track_positives(result, 1)
        else:
            track_positives(result, 0)
    # print results in form of precision and recall
    print("Precision :: Recall")
    print("Setosa: " + str(setosa_tp / setosa_ret) + " :: " + str(setosa_tp / 10))
    print("Versicolor: " + str(versicolor_tp / versicolor_ret) + " :: " + str(versicolor_tp / 10))
    print("Virginica: " + str(virginica_tp / virginica_ret) + " :: " + str(virginica_tp / 10))

#---------------Node Methods------------------------------------------------------------------------------------------
# nodes are represented by methods that fire activation functions and return 0 or 1
def node1(flower):
    global weights1
    # weight[4] is the threshold, bias is assumed at one
    activation = flower[0]*weights1[0]+flower[1]*weights1[1]+flower[2]*weights1[2]+flower[3]*weights1[3]-\
                 float(weights1[4])
    if activation > 0:
        return 1
    else:
        return 0


def node2(flower):
    global weights2
    activation = flower[0]*weights2[0]+flower[1]*weights2[1]+flower[2]*weights2[2]+flower[3]*weights2[3]-weights2[4]
    if activation > 0:
        return 1
    else:
        return 0


def node3(flower):
    global weights3
    activation = flower[0]*weights3[0]+flower[1]*weights3[1]+flower[2]*weights3[2]+flower[3]*weights3[3]-weights3[4]
    if activation > 0:
        return 1
    else:
        return 0

# ---------------Weight Change Methods-------------------------------------------------------------------------------
# each node (corresponding number) has its own increase and decrease methods for ease of visualization
def increase1(flower):
    global weights1
    global c
    weights1[0] = weights1[0] + c * flower[0]
    weights1[1] = weights1[1] + c * flower[1]
    weights1[2] = weights1[2] + c * flower[2]
    weights1[3] = weights1[3] + c * flower[3]
    weights1[4] = weights1[4] + c


def decrease1(flower):
    global weights1
    global c
    weights1[0] = weights1[0] - c * flower[0]
    weights1[1] = weights1[1] - c * flower[1]
    weights1[2] = weights1[2] - c * flower[2]
    weights1[3] = weights1[3] - c * flower[3]
    weights1[4] = weights1[4] - c


def increase2(flower):
    global weights2
    global c
    weights2[0] = weights2[0] + c * flower[0]
    weights2[1] = weights2[1] + c * flower[1]
    weights2[2] = weights2[2] + c * flower[2]
    weights2[3] = weights2[3] + c * flower[3]
    weights2[4] = weights2[4] + c


def decrease2(flower):
    global weights2
    global c
    weights2[0] = weights2[0] - c * flower[0]
    weights2[1] = weights2[1] - c * flower[1]
    weights2[2] = weights2[2] - c * flower[2]
    weights2[3] = weights2[3] - c * flower[3]
    weights2[4] = weights2[4] - c


def increase3(flower):
    global weights3
    global c
    weights3[0] = weights3[0] + c * flower[0]
    weights3[1] = weights3[1] + c * flower[1]
    weights3[2] = weights3[2] + c * flower[2]
    weights3[3] = weights3[3] + c * flower[3]
    weights3[4] = weights3[4] + c


def decrease3(flower):
    global weights3
    global c
    weights3[0] = weights3[0] - c * flower[0]
    weights3[1] = weights3[1] - c * flower[1]
    weights3[2] = weights3[2] - c * flower[2]
    weights3[3] = weights3[3] - c * flower[3]
    weights3[4] = weights3[4] - c

# --------------------------------------------------------------------------------------------------------------------

# This method finds the first 1 in the list and zeros the rest, so that it can displayed easily, and catches a few
# errors. The it compares them and responds with whether they match or not
def test_output(results, flower):
    for i in range(3):
        if results[i] == 1:
            results[(i+1) % 3] = 0
            results[(i+2) % 3] = 0
            break
    if results == flower:
        return 1
    else:
        return 0

#--------------------Modeling Attempt One-----------------------------------------------------------------------------
# this is my second attempt at using a python modelling tool, apparantly the binaries of the library cannot be used on
# CPU or something, took like four hours to figure out to I left it in.
def modeling_tool():

    train_x, train_labels = input_training_set()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    estimator = tf.estimator.LinearClassifier(feature_columns=my_feature_columns, n_classes=3)
    estimator.train(input_fn=input_training_set, steps=1)
    eval_result = estimator.evaluate(input_fn=input_testing_set)
    print(eval_result)


def input_training_set():
    global flowers_train
    att_a = np.empty(120)
    att_b = np.empty(120)
    att_c = np.empty(120)
    att_d = np.empty(120)
    labels = np.empty(120, dtype=int)
    i = 0
    for flower in flowers_train:
        att_a[i] = flower[0]
        att_b[i] = flower[1]
        att_c[i] = flower[2]
        att_d[i] = flower[3]
        if flower[4] == [1, 0, 0]:
            labels[i] = int(0)
        elif flower[4] == [0, 1, 0]:
            labels[i] = int(1)
        else:
            labels[i] = int(2)
        i += 1

    features = {'SepalLength': att_a,
                'SepalWidth': att_b,
                'PetalLength': att_c,
                'PetalWidth': att_d}
    return features, labels


def input_testing_set():
    global flowers_test
    att_a = np.empty(30)
    att_b = np.empty(30)
    att_c = np.empty(30)
    att_d = np.empty(30)
    labels = np.empty(30, dtype=int)

    i = 0
    for flower in flowers_test:
        att_a[i] = flower[0]
        att_b[i] = flower[1]
        att_c[i] = flower[2]
        att_d[i] = flower[3]
        if flower[4] == [1, 0, 0]:
            classifier = 0
            labels[i] = classifier
        elif flower[4] == [0, 1, 0]:
            classifier = 1
            labels[i] = classifier
        else:
            classifier = 2
            labels[i] = classifier
        i += 1

    features = {'SepalLength': att_a,
                'SepalWidth': att_b,
                'PetalLength': att_c,
                'PetalWidth': att_d}
    return features, labels

# --------------------------------------------------------------------------------------------------------------

# Track true positives as well as combined positives for each class to do precision and recall
def track_positives(results, correct):
    global setosa_ret
    global setosa_tp
    global versicolor_ret
    global versicolor_tp
    global virginica_ret
    global virginica_tp
    if results == [0, 0, 0]:
        return
    if results == [1, 0, 0]:
        if correct == 1:
            setosa_tp += 1
        setosa_ret += 1
    if results == [0, 1, 0]:
        if correct == 1:
            versicolor_tp += 1
        versicolor_ret += 1
    if results == [0, 0, 1]:
        if correct == 1:
            virginica_tp += 1
        virginica_ret += 1

# Tests against all the flowers and makes a text document
def testall_and_text():
    global flowers_train
    global flowers_test
    global setosa_ret
    global setosa_tp
    global versicolor_ret
    global versicolor_tp
    global virginica_ret
    global virginica_tp
    global weights1
    global weights2
    global weights3
    global sweights1
    global sweights2
    global sweights3
    f = open("TestResults.txt", "w+")
    setosa_ret = 0
    setosa_tp = 0
    versicolor_ret = 0
    versicolor_tp = 0
    virginica_ret = 0
    virginica_tp = 0
    correct = 0
    incorrect = 0
    f.write("Setosa = [1,0,0]\tVersicolor = [0,1,0]\tVirginica = [0,0,1]\n\n------------------------------------------" \
            "--------------------------------\n")
    for flower3 in flowers_train:
        result = [node1(flower3), node2(flower3), node3(flower3)]
        if test_output(result, flower3[4]):
            track_positives(result, 1)
            string = "correct"
            correct += 1
        else:
            track_positives(result, 0)
            string = "incorrect"
            incorrect += 1
        f.write("[" + str(result[0]) + ',' + str(result[1]) + "," + str(result[2]) + "] : [" + str(flower3[4][0]) \
                + ',' + str(flower3[4][1]) + ',' + str(flower3[4][2]) + "] : " + string + "\n")
    for flower3 in flowers_test:
        result = [node1(flower3), node2(flower3), node3(flower3)]
        if test_output(result, flower3[4]):
            track_positives(result, 1)
            string = "correct"
            correct += 1
        else:
            track_positives(result, 0)
            string = "incorrect"
            incorrect += 1
        f.write("[" + str(result[0]) + ',' + str(result[1]) + "," + str(result[2]) + "] : [" + str(flower3[4][0]) \
                + ',' + str(flower3[4][1]) + ',' + str(flower3[4][2]) + "] : " + string + "\n")
    f.write("Correct : " + str(correct) + "\n")
    f.write("Incorrect : " + str(incorrect) + "\n")
    f.write("Error : " + str(incorrect/(correct+incorrect)) + "\n\n")
    f.write("Starting Weights: " + str(sweights1) + " : " + str(sweights2) + " : " + str(sweights3) + "\n")
    f.write("Final Weights: " + str(weights1) + " : " + str(weights2) + " : " + str(weights3) + "\n")
    f.write("--------------------------------------------------------------------------\n")
    f.write("Precision :: Recall\n")
    f.write("Setosa: " + str(setosa_tp / setosa_ret) + " :: " + str(setosa_tp / 50) + "\n")
    f.write("Versicolor: " + str(versicolor_tp / versicolor_ret) + " :: " + str(versicolor_tp / 50) + "\n")
    f.write("Virginica: " + str(virginica_tp / virginica_ret) + " :: " + str(virginica_tp / 50) + "\n")
    f.close()

# This is where the methods are called from
set_weights()
train_model()
# testall_and_text()




