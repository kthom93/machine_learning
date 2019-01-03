from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import copy as copy

class PerceptronLearner(SupervisedLearner):
    weights = [] #A 2 dimensional array with each row associated with a perceptron
    numberRows = 0 #The number of different possible label values

    def __init__(self):
        pass

    def train(self, features, labels):
        learning_rate = 0.9 #The learning rate of the Perceptron
        self.numberRows = labels.value_count(0)
        counter = 0 #Determines how many times in a row to run the perceptron without any increase in accuracy
        prev_acc = 0 #stores the previous epochs accuracy
        outputs = labels.value_count(0) #Number of possible outputs
        self.weights = np.zeros((outputs, features.cols + 1)) #createing the 2 dimensional weight array
        total_epochs = 0 #Tracks how many epochs the program runs
        
        #k represents what Perceptron we are on and is related to the output
        for k in range(outputs):
            counter = 0
            while counter < 10: #program will run this number of times without an increase in accuracy
                features.shuffle(labels); #shuffle the data

                #runs through all the data
                for i in range(len(features.data)):
                    #this is the delta w array
                    weight_change = np.zeros(features.cols + 1)
                    #the input array plus a bias
                    inputs = copy.deepcopy(features.data[i])
                    inputs.append(1)
                    #calculate the net
                    net = (self.weights[k] * inputs).sum()
                    output = 0
                    if net > 0:
                        output = 1
                    for j in range(len(weight_change)):
                        target = 0
                        if k == labels.data[i][0]:
                            target = 1
                        weight_change[j] = learning_rate * (target - output) * inputs[j]
                    self.weights[k] = self.weights[k] + weight_change
                curr_acc = self.test_accuracy(features, labels)
                total_epochs += 1
                if prev_acc < curr_acc:
                        counter = 0
                        prev_acc = curr_acc
                else:
                    counter += 1
        print("Weight:  ", self.weights)
        print("Total Epochs:  ", total_epochs)


    def test_accuracy(self, features, labels):
        correct = 0
        incorrect = 0
        for i in range(labels.value_count(0)):
            for j in range(len(features.data)):
                inputs = copy.deepcopy(features.data[j])
                inputs.append(1)
                net = (self.weights[i] * inputs).sum()
                if net > 0 and i == labels.data[j][0]:
                    correct += 1
                elif net > 0 and i != labels.data[j][0]:
                    incorrect += 1
                elif net <= 0 and i != labels.data[j][0]:
                    correct += 1
                elif net <= 0 and i == labels.data[j][0]:
                    incorrect += 1
                else:
                    incorrect += 1
                    
        if incorrect == 0:
            return 1
        else: 
            return correct/(incorrect + correct)
        



    def predict(self, features, labels):
        for i in range(self.numberRows):
            inputs = copy.deepcopy(features)
            inputs.append(1)
            net = (self.weights[i] * inputs).sum()
            output = []
            if net > 0:
                output = [i]
                del labels[:]
                labels += output
                return
        labels += [0]


