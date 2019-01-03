from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import copy as copy
import sys

class InstanceBasedLearner(SupervisedLearner):
    k = 7
    limit = 10000
    weighted = True
    regression = False
    limitData = True
    fillDistance = 0.5
    outputsCount = 0
    featuresCount = 0
    attributesCount = 0
    features = []
    labels = []
    featuresValueCounts = []
    
    def mhd(self, first, second):
        distance = 0
        for i in range(len(first)):
            if self.featuresValueCounts[i] == 0:
                if first[i] == np.inf or second[i] == np.inf:
                    distance += self.fillDistance
                else:
                    distance += abs(first[i] - second[i])
            else:
                if first[i] != second[i]:
                    distance += self.fillDistance
        return distance

    def calcDistances(self, testFeature):
        distances = []
        for i in range(len(self.features)):
                distances.append(self.mhd(testFeature, self.features[i]))
        return distances

    def findNeighbors(self, distances):
        nearestNeighbors = []
        while len(nearestNeighbors) < self.k:
            minIndex = distances.index(min(distances))
            if distances[minIndex] != 0:
                nearestNeighbors.append(minIndex)
            distances[minIndex] = np.inf
        return nearestNeighbors

    def calcWeights(self, nearestNeighbors, distances):
        weights = []
        for i in range(self.k):
            weight = 0
            if distances[nearestNeighbors[i]] != 0:
                weight = 1 / (distances[nearestNeighbors[i]]**2)
            weights.append(weight)
        return weights

    def calcNonWeightedLabel(self, nearestNeighbors):
        neighborList = []
        for i in range(self.k):
            neighborList.append(self.labels[nearestNeighbors[i]][0])
        return max(set(neighborList), key=neighborList.count)

    def calcWeightedLabel(self, nearestNeighbors, distances):
        neighborListWeights = [0] * self.outputsCount
        for i in range(self.k):
            distance = distances[nearestNeighbors[i]]
            weightedDistance = 0
            if distance != 0:
                weightedDistance = (1 / distance**(2))
            neighborListWeights[int(round(self.labels[nearestNeighbors[i]][0]))] += weightedDistance
        return neighborListWeights.index(max(neighborListWeights))

    def calcRegNonWeightedLabel(self, nearestNeighbors):
        total = 0
        for i in range(self.k):
            total += self.labels[nearestNeighbors[i]][0]
        return total/self.k

    def calcRegWeightedLabel(self, nearestNeighbors, distances):
        weights = self.calcWeights(nearestNeighbors, distances)
        label = 0
        for i in range(self.k):
            label =  label + (self.labels[nearestNeighbors[i]][0] * weights[i])
        return label / np.sum(weights)

    def calcRegressionLabel(self, nearestNeighbors, distances):
        if self.weighted:
            return self.calcRegWeightedLabel(nearestNeighbors, distances)
        else:
            return self.calcRegNonWeightedLabel(nearestNeighbors)

    def calcLabel(self, nearestNeighbors, distances):
        if self.weighted:
            return self.calcWeightedLabel(nearestNeighbors, distances)
        else:
            return self.calcNonWeightedLabel(nearestNeighbors)

    def getFeaturesValueCounts(self, features):
        for i in range(self.attributesCount):
            self.featuresValueCounts.append(features.value_count(i))

    def getFeatures(self, features):
        if self.limitData:
            for i in range(self.limit):
                self.features.append(features[i])
        else:
            self.features = features

    def getLabels(self, labels):
        if self.limitData:
            for i in range(self.limit):
                self.labels.append(labels[i])
        else:
            self.labels = labels

    def train(self, features, labels):
        print("k = ", self.k)
        if self.limit > len(features.data):
            self.limit = len(features.data)
        features.shuffle(labels)
        self.getFeatures(features.data)
        self.getLabels(labels.data)
        self.outputsCount = labels.value_count(0)
        self.attributesCount = len(features.data[0])
        self.featuresCount = len(features.data)
        self.getFeaturesValueCounts(features)
        print("Limit:  ", self.limit)
        self.test()
    
    def test(self):
        pass

    def predict(self, features, labels):
        del labels[:]
        distances = self.calcDistances(features)
        nearestNeighbors = self.findNeighbors(copy.copy(distances))
        if self.regression:
            labels += [self.calcRegressionLabel(nearestNeighbors, distances)]
        else:
            labels += [self.calcLabel(nearestNeighbors, distances)]
