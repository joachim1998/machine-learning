"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary
from matplotlib import pyplot as plt


# (Question 2)

def get_sets(nb_samples, nb_training_set, seed, which):
    """
    """
    if which == 1:
        dataset = make_data1(nb_samples, random_state = seed)
    else:
        dataset = make_data2(nb_samples, random_state = seed)

    proportion_training_set = nb_training_set/nb_samples

    return train_test_split(dataset[0], dataset[1], train_size=proportion_training_set, test_size=1-proportion_training_set , random_state=seed)

def get_accuracy(n_neighbors, seed, which, dataset_size, trainingSet_size):
    """
    """
    #get the sets
    x_train_sample, x_test_sample, y_train_sample, y_test_sample = get_sets(dataset_size, trainingSet_size, seed, which)

    for i in range(len(n_neighbors)):
        #get the KN neighbors for each nb_neighbors
        knn = KNeighborsClassifier(n_neighbors=n_neighbors[i]).fit(x_train_sample, y_train_sample)

        #we predict from the training samples
        prediction = knn.predict(x_test_sample)

        #compute the accuracy
        accuracy = accuracy_score(y_test_sample, prediction)

        #plot
        fname = "KNN=" + str(n_neighbors[i]) + "_ds=" + str(which)
        title = "KNN of " + str(n_neighbors[i]) \
                 + " neighbours and with an accuracy of %0.4f" %accuracy

        plot_boundary(fname, knn, x_test_sample, y_test_sample, 0.1, title)

        print("The accuracy for the dataset " + str(which) + " is : %0.4f" %accuracy)

def tenfold(nb_sub, nb_neighbors, nb_samples, which):
    """
    """
    results = []
    neighbors_toplot = []
    optimal_nb_neighbors = -1
    max_score = -1
    neighbors = 1

    if which == 1:
        dataset = make_data1(nb_samples, nb_sub)
    else:
        dataset = make_data2(nb_samples, nb_sub)

    #Ten-fold cross validation strategy
    while neighbors <= nb_neighbors:
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        scores = cross_val_score(knn, dataset[0], dataset[1], cv=nb_sub, scoring='accuracy')
        mean_score = scores.mean()
        results.append(mean_score)
        neighbors_toplot.append(neighbors)

        #Determining the optimal number of neighbors
        if mean_score > max_score:
            max_score = mean_score
            optimal_nb_neighbors = neighbors

        neighbors += 1

    print("The optimal number of neighbors is : " + str(optimal_nb_neighbors) + \
            " with and accuracy of %0.4f" %max_score)

    plt.plot(neighbors_toplot, results)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    file_name = "Tenfold_cross_ds=" + str(which)
    plt.savefig("%s.pdf" %file_name)

if __name__ == "__main__":

    dataset_size = 2000
    trainingSet_size = 150
    n_neighbors = [1, 5, 10, 75, 100, 150]
    seed = 1

    #compute the accuracy
    for i in range(2):
        get_accuracy(n_neighbors, seed, i+1, dataset_size, trainingSet_size)

        #use the ten-fold cross validation startegy
        tenfold(10, n_neighbors[5], dataset_size, i+1)