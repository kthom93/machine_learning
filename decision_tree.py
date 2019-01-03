from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import copy as copy
import sys
import random

class DecisionTreeLearner(SupervisedLearner):
    root = None
    outputsCount = 0
    featuresCount = []
    pruneAccuracy = 0.9
    leafNodeCount = 0
    innerNodeCount = 0
    depth = 0

    class Node:
        usedFeatures = []
        children = []
        splitFeatureIndex = None
        splitFeatureValue = None
        assignedLabel = None
        leafNode = False
        features = []
        labels = []
        def __init__(self):
            usedFeatures = []
            children = []
            splitFeatureIndex = None
            splitFeatureValue = None
            assignedLabel = None
            leafNode = False
            features = []
            labels = []
      
    def __init__(self):
        pass

    def printNode(self, node):
        print()
        print("---------------------")
        print("Used Features Array: ", node.usedFeatures)
        print("Children Array:  ", node.children)
        print("Split Feature Index:  ", node.splitFeatureIndex)
        print("Split Feature Value:  ", node.splitFeatureValue)
        print("AssignedLabel:  ", node.assignedLabel)
        print("Leaf Node:  ",  node.leafNode)
        print("Table Size:  ", len(node.features))
        print("Check Size:  ", len(node.labels))
        print("---------------------")
        print()

    def printTree(self, node):
        for i in range(len(node.children)):
            self.printNode(node.children[i])
        for i in range(len(node.children)):
            self.printTree(node.children[i])

    def initializeData(self, features, labels):
        self.root = None
        self.outputsCount = 0
        self.featuresCount = []
        updatedFeatures = self.fillUnknownAttributes(features)
        self.outputsCount = labels.value_count(0)
        for i in range(features.cols):
            self.featuresCount.append(1)
        for i in range(len(features.data)):
            for j in range(len(updatedFeatures[i])):
                if self.featuresCount[j] < updatedFeatures[i][j] + 1:
                    self.featuresCount[j] = int(round(updatedFeatures[i][j] + 1))
        self.root = DecisionTreeLearner.Node()
        self.root.usedFeatures = []
        self.root.children = []
        self.root.splitFeatureIndex = None
        self.root.splitFeatureValue = None
        self.root.assignedLabel = None
        self.root.leafNode = False
        self.root.features = []
        self.root.labels = []
        self.root.features = updatedFeatures
        self.root.labels = labels.data
        for i in range(len(updatedFeatures[0])):
            self.root.usedFeatures.append(False)
        self.assignLabel(self.root)

    def calcInfoLogPart(self, divisor, topNums):
        sum = 0;
        if divisor != 0:
            for i in range(len(topNums)):
                if topNums[i] != 0:
                    sum += ((topNums[i] / divisor) * np.log2(topNums[i] / divisor))
        return -sum

    def calcInfo(self, index, node):
        total = len(node.labels)
        featuresTotal = []
        insideTotals = []
        for i in range(self.outputsCount):
            featuresTotal.append(0)
            insideTotals.append(0)
        for i in range(len(node.features)):
            featuresTotal[int(round(node.features[i][index]))] += 1
        topNums = []
        for i in range(self.featuresCount[index]):
            temp = []
            for j in range(self.outputsCount):
                temp.append(0)
                for k in range(len(node.features)):
                    if node.labels[k][0] == j:
                        if node.features[k][index] == i:
                            temp[j] += 1
            topNums.append(temp)
        outputSum = 0
        for i in range(self.featuresCount[index]):
            outputSum += ((featuresTotal[i] / total) * (self.calcInfoLogPart(featuresTotal[i], topNums[i])))
        return outputSum

    def assignLabel(self, node):
        labelCount = []
        for i in range(self.outputsCount):
            labelCount.append(0)
            for j in range(len(node.labels)):
                if node.labels[j][0] == i:
                    labelCount[i] += 1
        node.assignedLabel = labelCount.index(max(labelCount))

    def decisionTreeRecursive(self, node):
        info = []
        proceed = False
        for i in range(len(node.usedFeatures)):
            if node.usedFeatures[i] == False:
                proceed = True
                #info.append(self.calcInfo(i, node))
                info.append(random.randint(-20,20))
            else:
                info.append(100)
        small = min(info)
        for i in range(len(node.usedFeatures)):
            if info[i] == small:
                if node.usedFeatures[i] == False:
                    node.splitFeatureIndex = i
        if proceed:
            for i in range(len(node.features)):
                newNode = True
                childValues = []
                for j in range(len(node.children)):
                    childValues.append(node.children[j].splitFeatureValue)
                if (node.features[i][node.splitFeatureIndex] in childValues):
                    newNode = False
                if newNode:
                    node.children.append(DecisionTreeLearner.Node())
                    node.children[-1].usedFeatures = []
                    node.children[-1].children = []
                    node.children[-1].splitFeatureIndex = None
                    node.children[-1].splitFeatureValue = None
                    node.children[-1].assignedLabel = None
                    node.children[-1].leafNode = False
                    node.children[-1].features = []
                    node.children[-1].labels = []
                    node.children[-1].splitFeatureValue = node.features[i][node.splitFeatureIndex]
                    node.children[-1].usedFeatures = copy.copy(node.usedFeatures)
                    node.children[-1].usedFeatures[node.splitFeatureIndex] = True
                    for j in range(len(node.features)):
                        if node.children[-1].splitFeatureValue == node.features[j][node.splitFeatureIndex]:
                            node.children[-1].features.append(node.features[j])
                            node.children[-1].labels.append(node.labels[j])

                    self.assignLabel(node.children[-1])
        for i in range(len(node.children)):
            self.decisionTreeRecursive(node.children[i])
                
    def fillUnknownAttributes(self, features):
        mostFrequentAttribute = []
        dividingTotals = []
        newFeatures = copy.deepcopy(features.data)

        for i in range(len(newFeatures[0])):
            mostFrequentAttribute.append(0)
            dividingTotals.append(0)
            for j in range(len(newFeatures)):
                if newFeatures[j][i] != np.inf:
                    mostFrequentAttribute[i] += newFeatures[j][i]
                    dividingTotals[i] += 1

        for i in range(len(newFeatures[0])):
            for j in range(len(newFeatures)):
                if newFeatures[j][i] == np.inf:
                    newFeatures[j][i] = round(mostFrequentAttribute[i] / dividingTotals[i]) 
        return newFeatures

    def pruneTree(self, node):
        if self.calcNodeAccuracy(node) > self.pruneAccuracy:
            node.children = []
            return
        for i in range(len(node.children)):
            self.pruneTree(node.children[i])

    def calcNodeAccuracy(self, node):
        right = 0
        wrong = 0
        for i in range(len(node.labels)):
            if node.labels[i][0] == node.assignedLabel:
                right += 1
            else:
                wrong +=1
        return right/ (right + wrong)

    def getTreeCounts(self, node, depth):
        depth += 1
        if depth > self.depth:
            self.depth = depth
        if len(node.children) == 0:
            self.leafNodeCount += 1
        else:
            self.innerNodeCount += 1
        for i in range(len(node.children)):
            self.getTreeCounts(node.children[i], depth)

    def train(self, features, labels):
        self.initializeData(features, labels)
        self.decisionTreeRecursive(self.root)
        self.leafNodeCount = 0
        self.innerNodeCount = 0
        self.depth = 0
        self.getTreeCounts(self.root, 0)
        print("Leafs:  ", self.leafNodeCount)
        print("Inner Nodes:  ", self.innerNodeCount)
        print("Depth:  ", self.depth)
        print()
        self.pruneTree(self.root)
        self.leafNodeCount = 0
        self.innerNodeCount = 0
        self.depth = 0
        self.getTreeCounts(self.root, 0)
        print("Pruned Tree")
        print("Leafs:  ", self.leafNodeCount)
        print("InnerNodes:  ", self.innerNodeCount)
        print("Depth:  ", self.depth)
        print()
        #self.printTree(self.root)

    def printTree(self, node):
        self.printNode(node)
        for i in range(len(node.children)):
            self.printTree(node.children[i])

    def transverseTree(self, node, features, labels):
        #print("Sanity Check")
        if len(node.children) == 0:
            labels += [node.assignedLabel]
        for i in range(len(node.children)):
            if features[node.splitFeatureIndex] == node.children[i].splitFeatureValue:
                self.transverseTree(node.children[i], features, labels)
        if len(labels) == 0:
            labels += [node.assignedLabel]

    def predict(self, features, labels):
        #print("_______________________________")
        del labels[:]
        self.transverseTree(self.root, features, labels)
