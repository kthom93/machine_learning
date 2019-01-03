from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import copy as copy
import sys

class NeuralNetLearner(SupervisedLearner):
    learning_rate = 0.3
    momentum = 0.5
    outputs = 0
    inputs = 0
    totalEpochs = 1000
    layers = 2
    outputNodes = []
    hiddenNodes = []
    useMomentum = True
    counter = 0
    bestAccuracy = 0
    mse = 0
    mse2 = 0

    class Node:
        def __init__(self):
            weights = []
            deltaWeights = []
            prevDeltaWeights = []
            output = 0
            error = 0
      
    def __init__(self):
        pass
    
    def activationFunction(self, net):
        return 1 / (1 + np.exp(-net))

    def calculateNet(self, nodeWeights, inputs):
        data = [1] + copy.deepcopy(inputs)
        return (nodeWeights * data).sum()

    def calculateOutputError(self, target, output):
        return (target - output) * output * (1 - output)

    def calculateDeltaWeight(self, node, prevOutput, index):
        momentum = 0
        if self.useMomentum:
            momentum = node.prevDeltaWeights[index] * self.momentum

        return ((self.learning_rate * node.error * prevOutput) + momentum)
    
    def calculateSummation(self, layer, index):
        total = 0
        if layer == 1:
            for i in range(self.outputs):
                total += (self.outputNodes[i].error * self.outputNodes[i].weights[index + 1])
        else:
            for i in range(self.inputs):
                total += (self.hiddenNodes[layer + 1][i].error * self.hiddenNodes[layer + 1][i].weights[index + 1])
        return total

    def calculateHiddenError(self, output, layer, index):
        upLayerSummation = self.calculateSummation(layer, index)
        return (output * (1 - output) * upLayerSummation)

    def generateWeights(self, number):
        return np.random.uniform(low = -1.0, high = 1.0, size = (number + 1,))

    def printNodes(self):
        print("Output Array")
        for i in range(self.outputs):
            print("Output Node: ", i)
            print("Weights: ", self.outputNodes[i].weights)
            print("Delta: ", self.outputNodes[i].deltaWeights)
            print("Output: ", self.outputNodes[i].output)
            print("Error: ", self.outputNodes[i].error)
            print()
        print("--------------------------")
        print("Hidden Layers")
        print()
        for i in reversed(xrange(2)):
            print("Layer  ", i + 1)
            print()
            for j in range(self.inputs):
                print("Hidden Node: ", j + 1)
                print("Weights: ", self.hiddenNodes[i][j].weights)
                print("Delta: ", self.hiddenNodes[i][j].deltaWeights)
                print("Output: ", self.hiddenNodes[i][j].output)
                print("Error: ", self.hiddenNodes[i][j].error)
                print()
            print("-----------------------")
        print("End of Data")


    def getInputs(self, layer, data):
        if layer == 0:
            return data
        inputs = []
        for i in range(self.inputs):
            inputs.append(self.hiddenNodes[layer - 1][i].output)
        return inputs

    def calculateWeightChangeList(self, node, layer):
        for i in range(self.outputs):
            self.outputNodes[i].prevDeltaWeights = self.outputNodes[i].deltaWeights
        for i in range(2):
            for j in range(self.inputs):
                self.hiddenNodes[i][j].prevDeltaWeights = self.hiddenNodes[i][j].deltaWeights

        for i in range(self.inputs + 1):
            if i == 0:
                node.deltaWeights[i] = self.calculateDeltaWeight(node, 1, i)
            else:
                node.deltaWeights[i] = self.calculateDeltaWeight(node, self.hiddenNodes[layer - 1][i - 1].output, i)


    def initialize(self):
        #print()
        #print("Initializing")
        #print()
        for i in range(self.outputs):
            self.outputNodes.append(NeuralNetLearner.Node())
            self.outputNodes[i].weights = self.generateWeights(self.inputs)
            self.outputNodes[i].deltaWeights = np.zeros(self.inputs + 1)
            self.outputNodes[i].prevDeltaWeights = np.zeros(self.inputs + 1)
            self.outputNodes[i].output = 0
            self.outputNodes[i].error = 0
        layerOne = []
        layerTwo = []
        layerThree = []
        layerFour = []
        for i in range(self.inputs):
            layerOne.append(NeuralNetLearner.Node())
            layerTwo.append(NeuralNetLearner.Node())
            layerThree.append(NeuralNetLearner.Node())
            layerFour.append(NeuralNetLearner.Node())
            layerOne[i].weights = self.generateWeights(self.inputs)
            layerOne[i].deltaWeights = np.zeros(self.inputs + 1)
            layerOne[i].prevDeltaWeights = np.zeros(self.inputs + 1)
            layerOne[i].output = 0
            layerOne[i].error = 0
            layerTwo[i].weights = self.generateWeights(self.inputs)
            layerTwo[i].deltaWeights = np.zeros(self.inputs + 1)
            layerTwo[i].deltaWeights = np.zeros(self.inputs + 1)
            layerTwo[i].output = 0
            layerTwo[i].error = 0 
            layerThree[i].weights = self.generateWeights(self.inputs)
            layerThree[i].deltaWeights = np.zeros(self.inputs + 1)
            layerThree[i].deltaWeights = np.zeros(self.inputs + 1)
            layerThree[i].output = 0
            layerThree[i].error = 0
            layerFour[i].weights = self.generateWeights(self.inputs)
            layerFour[i].deltaWeights = np.zeros(self.inputs + 1)
            layerFour[i].deltaWeights = np.zeros(self.inputs + 1)
            layerFour[i].output = 0
            layerFour[i].error = 0
        self.hiddenNodes.append(layerOne)
        self.hiddenNodes.append(layerTwo)
        self.hiddenNodes.append(layerThree)
        self.hiddenNodes.append(layerFour)

    def backpropagation(self, features, labels):
        features.shuffle(labels)
        for i in range(len(features.data)):
            for j in range(self.layers):
                for k in range(self.inputs):
                    inputs = self.getInputs(j, features.data[i])
                    net = self.calculateNet(self.hiddenNodes[j][k].weights, inputs)
                    self.hiddenNodes[j][k].output = self.activationFunction(net)
            for j in range(self.outputs):
                inputs = self.getInputs(self.layers, features.data[i])
                net = self.calculateNet(self.outputNodes[j].weights, inputs)
                self.outputNodes[j].output = self.activationFunction(net)
                self.mse += self.calcMSE(labels)
                if i < 5:
                    self.mse2 += self.calcMSE(labels)
            self.getDeltaWeights(labels.data[i][0])
            self.updateWeights()
            #print("Converge: ", self.outputNodes[0].weights)

    def updateWeights(self):
        for i in range(self.outputs):
            self.outputNodes[i].weights = self.outputNodes[i].weights + self.outputNodes[i].deltaWeights
        for i in range(2):
            for j in range(self.inputs):
                self.hiddenNodes[i][j].weights = self.hiddenNodes[i][j].weights + self.hiddenNodes[i][j].deltaWeights

    def getDeltaWeights(self, label):
        for i in range(self.outputs):
            target = 0
            if i == label:
                target = 1
            self.outputNodes[i].error = self.calculateOutputError(target, self.outputNodes[i].output)
            self.calculateWeightChangeList(self.outputNodes[i], 2)
        
        for i in reversed(xrange(2)):
            for j in range(self.inputs):
                self.hiddenNodes[i][j].error = self.calculateHiddenError(self.hiddenNodes[i][j].output, i, j)
                self.calculateWeightChangeList(self.hiddenNodes[i][j], i)

    def test(self):
        pass

    def calcMSE(self, labels):
        average = 0
        for j in range(self.outputs):
            target = 0
            if labels.data[j][0] == j:
                target = 1
            average += (target - self.outputNodes[j].output)
        return average / self.outputs
                

    def train(self, features, labels):
        #orig_stdout = sys.stdout
        #f = open('data.csv', 'w')
        #sys.stdout = f
        self.outputs = labels.value_count(0)
        self.inputs = features.cols
        #print("Inputs: ", self.inputs)
        #print("Outputs: ", self.outputs)
        self.initialize()
        #self.printNodes()
        #print()
        #print("Training")
        #print()

        number = 1
        while(self.counter < self.totalEpochs):
            #print("Epoch: ", number)
            self.mse = 0
            number += 1
            self.backpropagation(features, labels)
            currAccuracy = self.measure_accuracy(features, labels) 
            #print(number, "," , self.mse / len(features.data), ",",  self.mse2 / 5)
            print(number, ",", currAccuracy)
            if currAccuracy > self.bestAccuracy:
                self.counter = 0
                self.bestAccuracy = currAccuracy
            else:
                self.counter += 1

        #sys.stdout = orig_stdout
        #f.close()


        #self.testInitialize(features, labels)
        #testFeatures = [0.3, 0.7]
        #testLabels = [0.1, 1.0]
        #self.printNodes()
        #print()
        #print("Features:  ", testFeatures)
        #print("Labels:  ", testLabels)
        #self.testBackpropagation(testFeatures, testLabels)
        #self.printNodes()
        #self.testBackpropagation(testFeatures, testLabels)
        #self.printNodes()

    def checkAccuracy(self, features, labels):
        right = 0
        wrong = 0
        bestOutput = 0
        for i in range(len(features.data)):
            for j in range(self.layers):
                for k in range(self.inputs):
                    inputs = self.getInputs(j, features.data[i])
                    net = self.calculateNet(self.hiddenNodes[j][k].weights, inputs)
                    self.hiddenNodes[j][k].output = self.activationFunction(net)
            for k in range(self.outputs):
                inputs = self.getInputs(self.layers, features.data[i])
                net = self.calculateNet(self.outputNodes[k].weights, inputs)
                self.outputNodes[k].output = self.activationFunction(net)
            for i in range(self.outputs):
                if self.outputNodes[i].output > self.outputNodes[bestOutput].output:
                    bestOutput = i
            if bestOutput == labels.data[i][0]:
                right += 1
            else:
                wrong += 1
        if wrong == 0:
            return 1
        else:
            return right/(right + wrong)


    def predict(self, features, labels):
        #print("Features: ", features)
        del labels[:]
        bestOutput = 0
        for j in range(self.layers):
            for k in range(self.inputs):
                inputs = self.getInputs(j, features) #copy.deepcopy(features)
                net = self.calculateNet(self.hiddenNodes[j][k].weights , inputs)
                self.hiddenNodes[j][k].output =  self.activationFunction(net)
        for k in range(self.outputs):
            inputs = self.getInputs(self.layers, features)
            net = self.calculateNet(self.outputNodes[k].weights, inputs)
            self.outputNodes[k].output = self.activationFunction(net)
        
        for i in range(self.outputs):
            if self.outputNodes[i].output > self.outputNodes[bestOutput].output:
                bestOutput = i
            #print("Node: ", i + 1)
            #print("Output: ", self.outputNodes[i].output)
        #print("Prediction: ", bestOutput)
        #print("Prediction: ", bestOutput)
        labels += [bestOutput]


    def testBackpropagation(self, features, labels):
        print()
        print("Forward Propagation")
        print()
        for j in range(2):
            for k in range(self.inputs):
                print("Hidden Node Layer #", j + 1)
                print("Node #", k + 1)
                print("Node Weights: ", self.hiddenNodes[j][k].weights)
                inputs = self.getInputs(j, features)
                print("Inputs: ", inputs)
                net = self.calculateNet(self.hiddenNodes[j][k].weights , inputs)
                print("Net: ", net)
                self.hiddenNodes[j][k].output =  self.activationFunction(net)
                print("Output: ", self.hiddenNodes[j][k].output)
                print()
        for k in range(self.outputs):
            print("Output Node #", k)
            print("Node Weights: ", self.outputNodes[k].weights)
            inputs = self.getInputs(2, features)
            print("Inputs: ", inputs)
            net = self.calculateNet(self.outputNodes[k].weights, inputs)
            print("Net: ", net)
            self.outputNodes[k].output = self.activationFunction(net)
            print("Output: ", self.outputNodes[k].output)
            print()
        #self.printNodes()
        self.testGetDeltaWeights(labels)
        self.updateWeights()
        self.printNodes()

    def testGetDeltaWeights(self, labels):
        print()
        print("Back Propagation")
        print()
        for i in range(self.outputs):
            print("Output Node #", i + 1)
            target = labels[i]
            print("Target:  ", target)
            self.outputNodes[i].error = self.calculateOutputError(target, self.outputNodes[i].output)
            print("Error:  ", self.outputNodes[i].error)
            self.calculateWeightChangeList(self.outputNodes[i], 2)
            print("Delta Weights: ", self.outputNodes[i].deltaWeights)
            print()
        
        print()
        for i in reversed(xrange(2)):
            print("Hidden Layer #", i + 1)
            print()
            for j in range(self.inputs):
                print("Hidden Node #", j + 1)
                self.hiddenNodes[i][j].error = self.calculateHiddenError(self.hiddenNodes[i][j].output, i, j)
                print("Error: ", self.hiddenNodes[i][j].error)
                self.calculateWeightChangeList(self.hiddenNodes[i][j], i)
                print("Delta Weights: ", self.hiddenNodes[i][j].deltaWeights)
                print()


    def testInitialize(self, features, labels):
        print()
        print("Initializing Test Case")
        print()
        self.outputs = 2
        self.inputs = 2
        #  Initialize Output Nodes
        for i in range(self.outputs):
            self.outputNodes.append(NeuralNetLearner.Node())
            self.outputNodes[i].weights = self.generateWeights(self.inputs)
            self.outputNodes[i].deltaWeights = np.zeros(self.inputs + 1)
            self.outputNodes[i].output = 0
            self.outputNodes[i].error = 0
        self.outputNodes[0].weights[0] = 0.2
        self.outputNodes[0].weights[1] = -0.1
        self.outputNodes[0].weights[2] = 0.3
        self.outputNodes[1].weights[0] = 0.1
        self.outputNodes[1].weights[1] = -0.2
        self.outputNodes[1].weights[2] = -0.3

        #  Initialize Hidden Layer Nodes
        layerOne = []
        layerTwo = []
        for i in range(self.inputs):
            layerOne.append(NeuralNetLearner.Node())
            layerTwo.append(NeuralNetLearner.Node())
            layerOne[i].weights = self.generateWeights(self.inputs)
            layerOne[i].deltaWeights = np.zeros(self.inputs + 1)
            layerOne[i].output = 0
            layerOne[i].error = 0
            layerTwo[i].weights = self.generateWeights(self.inputs)
            layerTwo[i].deltaWeights = np.zeros(self.inputs + 1)
            layerTwo[i].output = 0
            layerTwo[i].error = 0
        layerOne[0].weights[0] = 0.1        
        layerOne[0].weights[1] = 0.2
        layerOne[0].weights[2] = -0.1
        layerOne[1].weights[0] = -0.2
        layerOne[1].weights[1] = 0.3
        layerOne[1].weights[2] = -0.3
        layerTwo[0].weights[0] = 0.1
        layerTwo[0].weights[1] = -0.2
        layerTwo[0].weights[2] = -0.3
        layerTwo[1].weights[0] = 0.2
        layerTwo[1].weights[1] = -0.1
        layerTwo[1].weights[2] = 0.3
        self.hiddenNodes.append(layerOne)
        self.hiddenNodes.append(layerTwo)


