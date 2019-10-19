"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary

def get_sets(nb_samples, nb_training_set, seed, which):
    """
    """
    if which == 1:
        dataset = make_data1(nb_samples, random_state = seed)
    else:
        dataset = make_data2(nb_samples, random_state = seed)

    proportion_training_set = nb_training_set/nb_samples

    return train_test_split(dataset[0], dataset[1], train_size=proportion_training_set, test_size=1-proportion_training_set , random_state=seed)

# (Question 1)

if __name__ == "__main__":
    # Definition of the dataset set size, the training set size and the different depths of the decision tree
    dataset_size = 2000
    trainingSet_size = 150
    depth = [1, 2, 4, 8, None]

    # Construction of the decision tree, predictions with it and computation of its accuracy five times for each depth
    # The data from the decision tree with a certain depth and with the best accuracy are plotted
    for i in range(2):
        accuracies = []
        best_accuracy = 0

        for j in range(len(depth)):
            accuracies.append([])

        for j in range(5):
            # Get the sets (testing and training)
            x_train_sample, x_test_sample, y_train_sample, y_test_sample = get_sets(dataset_size, trainingSet_size, j, i+1)

            for k in range(len(depth)):
                # Get the decision tree from the training sample
                decisionTree = DecisionTreeClassifier(max_depth = depth[k], random_state = seed).fit(x_train_sample, y_train_sample)

                # Predictions done from the training samples
                prediction = decisionTree.predict(x_test_sample)

                # Compute the accuracy
                accuracy = accuracy_score(y_test_sample, prediction)
                accuracies[k].append(accuracy)

                # Plot the best accuracy
                if accuracy > best_accuracy:
                    to_plot = [decisionTree, x_test_sample, y_test_sample, accuracy]
                    best_accuracy = accuracy

                    if j == 4:
                        fname = "DTC_depth=" + str(depth[k]) + "_ds=" + str(i+1)
                        title = "Decision Tree Classifier with a depth of " + str(depth[k]) \
                                + " with an accuracy of %0.4f" %to_plot[3]

                        plot_boundary(fname, to_plot[0], to_plot[1], to_plot[2], 0.1, title)

        # Compute the average accuracies over 5 generations of the dataset
        for j in range(5):
            avg_accuracy = sum(accuracies[j])/5
            deviation = np.std(accuracies[j])

            print("From dataset %d :" %(i+1))
            print("Depth = " + str(depth[j]))
            print("Average accuracy = %0.4f" %avg_accuracy)
            print("Deviation = %0.4f" %deviation)
            print()
