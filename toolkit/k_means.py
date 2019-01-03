from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np
import copy as copy
import sys
import math
import random

class KMeansLearner(SupervisedLearner):
    k = 7
    limit = 0.1
    random = True
    useSSE = True
    includeLabels = True
    calcSilhouette = False
    specialInitialization = False
    reassignCentroids = True
    reassignSize = 10
    fillDistance = 1
    features = []
    centroids = []
    prevCentroids = []
    clusters = []
    assignedClusters = []
    numAttributes = 0
    attributesValueCounts = []
    silhouettes = []
    prevTotalSSE = 0

    def __init__(self):
        pass

    def printClusters(self):
        for i in range(self.k):
            print()
            print("Cluster:  ", i)
            print("Length:  ", len(self.clusters[i]))
            print("SSE:  ", self.calcSSE(i))
            #for j in range(len(self.clusters[i])):
            #    print(self.clusters[i][j])
            print("---------------------")
            print()

    def printAssignedClusters(self):
        print()
        print("Assigned Clusters")
        print()
        for i in range(len(self.assignedClusters)):
            print(i, "=", self.assignedClusters[i])
        print("-----------------")
        print()

    def printCentroids(self):
        print()
        print()
        print("Centroids")
        for i in range(self.k):
            print("Centroid:  ", i)
            print(self.centroids[i])
            print()

    def initialize(self, features, labels):
        self.features = copy.deepcopy(features.data)
        if self.includeLabels:
            for i in range(len(self.features)):
                self.features[i].append(labels.data[i][0])
        if self.calcSilhouette:
            for i in range(len(self.features)):
                self.silhouettes.append([])
        self.numAttributes = len(self.features[0])
        for i in range(len(features.data[0])):
            self.attributesValueCounts.append(features.value_count(i))
        if self.includeLabels:
            self.attributesValueCounts.append(labels.value_count(0))
        if self.specialInitialization:
            for i in range(self.k):
                self.centroids.append(copy.deepcopy(self.features[0]))
                self.clusters.append([])
            for i in range(len(self.features)):
                self.clusters[random.randint(0, self.k - 1)].append(self.features[i])
            self.calcNewCentroids()
        else:
            for i in range(self.k):
                self.centroids.append(copy.deepcopy(self.features[i]))
                self.clusters.append([])
    
    def euclideanDistance(self, first, second):
        distance = 0
        for i in range(len(first)):
            if self.attributesValueCounts[i] == 0:
                if first[i] == np.inf or second[i] == np.inf:
                    distance += np.square(self.fillDistance)
                else:
                    distance += np.square(first[i] - second[i])
            else:
                if first[i] == np.inf or second[i] == np.inf:
                    distance += np.square(self.fillDistance)
                elif first[i] != second[i]:
                    distance += np.square(self.fillDistance)
        return np.sqrt(distance)
    
    def calcTotalSSE(self):
        totalSSE = 0
        for i in range(self.k):
            totalSSE = totalSSE + self.calcSSE(i)
        return totalSSE
    
    def calcSSE(self, index):
        total = 0
        for i in range(len(self.clusters[index])):
            total += np.square(self.euclideanDistance(self.centroids[index], self.clusters[index][i]))
        return total

    def calcNewCentroids(self):
        self.prevCentroids = copy.deepcopy(self.centroids)
        for i in range(self.k):
            for j in range(self.numAttributes):
                if (self.attributesValueCounts[j] == 0):
                    total = 0
                    numberAdded = 0
                    for k in range(len(self.clusters[i])):
                        if (self.clusters[i][k][j] != np.inf):
                            total += self.clusters[i][k][j]
                            numberAdded += 1
                    if (numberAdded == 0):
                        self.centroids[i][j] = np.inf
                    else:
                        self.centroids[i][j] = total / numberAdded
                else:
                    counts = np.zeros(int(round(self.attributesValueCounts[j])))
                    for k in range(len(self.clusters[i])):
                        if (self.clusters[i][k][j] != np.inf):
                            index = int(round(self.clusters[i][k][j]))
                            counts[index] = counts[index] + 1
                    maxIndex = 0
                    for k in range(len(counts)):
                        if counts[k] > counts[maxIndex]:
                            maxIndex = k
                    self.centroids[i][j] = maxIndex

    def calcCentroidsMovement(self):
        total = 0
        for i in range(self.k):
            total = total + self.euclideanDistance(self.centroids[i], self.prevCentroids[i])
        print("Total Centroid Difference:  ", total)
        return total

    def calcDistances(self):
        distances = []
        for i in range(self.k):
            centroidDistances = []
            for j in range(len(self.features)):
                centroidDistances.append(self.euclideanDistance(self.centroids[i], self.features[j]))
            distances.append(centroidDistances)
        return distances

    def calcBiggestDifference(self):
        lengths = []
        for i in range(len(self.clusters)):
            lengths.append(len(self.clusters[i]))
        return max(lengths) - min(lengths)

    def calcAverageSilhouette(self):
        self.calcSilhouettes()
        return np.mean(self.silhouettes)

    def calcSilhouettes(self):
        for i in range(len(self.features)):
            a = self.a(i)
            b = self.b(i)
            self.silhouettes[i] = (b - a) / max(a, b)

    def a(self, index):
        total = 0
        for i in range(len(self.clusters[self.assignedClusters[index]])):
            total = total + self.euclideanDistance(self.features[index], self.clusters[self.assignedClusters[index]][i])
        return total / len(self.clusters[self.assignedClusters[index]])
    
    def b(self, index):
        total = np.zeros(self.k - 1)
        counter = 0
        for i in range(self.k):
            if self.assignedClusters[index] != i:
                for j in range(len(self.clusters[i])):
                    total[counter] = total[counter] + self.euclideanDistance(self.features[index], self.clusters[i][j])
                total[counter] = total[counter] / len(self.clusters[i])
                counter = counter + 1
        return min(total)

    def groupFeatures(self, distances):
        for i in range(self.k):
            self.clusters[i] = []
        self.assignedClusters = []
        for i in range(len(self.features)):
            test = []
            for j in range(self.k):
                test.append(distances[j][i])
            index = test.index(min(test))
            self.clusters[index].append(self.features[i])
            self.assignedClusters.append(index)

    def cluster(self):
        iterate = True
        iterations = 0

        while iterate:
            iterations = iterations + 1
            print("****************")
            print("Iteration: ", iterations)
            print("****************")
            print("Total Features:  ", len(self.features))
            self.printCentroids()
            distances = self.calcDistances()
            self.groupFeatures(distances)
            self.printClusters()
            #self.printAssignedClusters()
            totalSSE = self.calcTotalSSE()
            print("Total SSE:  ", totalSSE)
            self.calcNewCentroids()
            if (self.useSSE):
                if (abs(self.prevTotalSSE - totalSSE) < self.limit):
                    iterate = False
                else:
                    self.prevTotalSSE = totalSSE
            else: 
                if (self.calcCentroidsMovement() < self.limit):
                    iterate = False
        print("Iterations before convergence:  ", iterations)
        print("Biggest Difference in Group Sizes:  ", self.calcBiggestDifference())
        if self.calcSilhouette:
            print("Average Silhouette:  ", self.calcAverageSilhouette())
           
    def test(self):
        self.attributesValueCounts = [2, 2, 2]
        array1 = np.array([1, 1, 1])
        array2 = np.array([4, 4, 4])
        print(self.euclideanDistance(array1, array2))
        print(np.sqrt(27))

    def train(self, features, labels):
        if (self.random):
            features.shuffle(labels)
        print("Number of Clusters = ", self.k)
        self.initialize(features, labels)
        self.cluster()

    def predict(self, features, labels):
        del labels[:]
        labels += [0]
