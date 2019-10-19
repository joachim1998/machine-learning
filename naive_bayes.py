"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary

def get_sets(nb_samples, nb_training_set, seed, which):
    """
    Return the training and testing sets for a given number of samples, a proportion of training
    set, a seed and for a dataset
    
    Arguments:
        nb_sample: the number of samples in the dataset
        nb_training_set: size of the training set
        seed: the seed used to make some random operations
        which: which dataset should be used
        
    Return:
        The result of the function train_test_split on the part X and y of the dataset, the proportion
        of the training set and learning set and on the seed.
    """
    if which == 1:
        dataset = make_data1(nb_samples, random_state = seed)
    else:
        dataset = make_data2(nb_samples, random_state = seed)

    proportion_training_set = nb_training_set/nb_samples

    return train_test_split(dataset[0], dataset[1], train_size=proportion_training_set, test_size=1-proportion_training_set , random_state=seed)


class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        """Fit a Gaussian naive Bayes model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # Get all the unique classes
        self.__classes, indices = np.unique(y, return_inverse=True)

        # Probability of belonging to a given class
        self.__p_y = np.zeros(len(self.__classes))

        # Mean of the elements for all the classes
        self.__moy = []

        # Variance of the elements for all the classes
        self.__var = []
        
        # Matrix of sorted samples in the different classes
        attributes = []
        
        # Creation of the sublists corresponding to the different classes
        for i in range(len(self.__classes)):
            self.__moy.append([])
            self.__var.append([])
            attributes.append([])

        # Computing P(y)
        for i in indices:
            self.__p_y[i] += 1

        div = len(self.__p_y)

        for i in range(len(self.__p_y)):
            self.__p_y[i] /= div

        # Filling attributes with the samples
        for i in range(len(indices)):
            attributes[indices[i]].append(X[i])
            
        # Computation of the mean and the variance
        for i in range(len(attributes)):
            self.__moy[i] = np.mean(attributes[i], axis = 0)
            self.__var[i] = np.var(attributes[i], axis = 0)

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        
        p = self.predict_proba(X)
        y = self.__classes[np.argmax(p, axis = 1)]

        return y
                

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        
        p = []
        for h in range(len(X)):             # Number of loops = number of samples
            max = 0
            Z = 0
            p.append([])
            for i in range(len(self.__classes)):
                Py = self.__p_y[i]          # Product of the different probabilities for one class
                for j in range(len(X[0])):   # Number of loops = number of features
                    temp = X[h][j]-self.__moy[i][j]
                    exp_num = math.pow(temp, 2)
                    exp_den = 2*self.__var[i][j]
                    exp = math.exp(-(exp_num/exp_den))
                    temp = 2*math.pi*self.__var[i][j]
                    factor_den = math.pow(temp, 1/2)
                    Py *= (1/factor_den)*exp
                    
                p[h].append(Py)
                Z += Py
                
            for i in range(len(p[h])):
                p[h][i] /= Z
            
        p = np.matrix(p)
        
        return p

if __name__ == "__main__":
    
    dataset_size = 2000
    trainingSet_size = 150
                               
    for i in range(2):
        x_train_sample, x_test_sample, y_train_sample, y_test_sample = get_sets(dataset_size, trainingSet_size, 1, i+1)

        nb = GaussianNaiveBayes().fit(x_train_sample, y_train_sample)
        prediction = nb.predict(x_test_sample)
        accuracy = accuracy_score(y_test_sample, prediction)
                               
        fname = "NB_ds=" + str(i+1)
        title = "Naive Bayes classification with an accuracy of %0.4f" %accuracy

        plot_boundary(fname, nb, x_test_sample, y_test_sample, 0.1, title)
        
