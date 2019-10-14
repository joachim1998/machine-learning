"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data1, make_data2
from plot import plot_boundary


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
        classes, indices = np.unique(y, return_inverse=True)

        # Probability of belonging to a given class
        self.__p_y = np.zeros(len(classes))

        # Mean of the elements for all the classes
        self.__moy = np.zeros(len(classes),2)

        # Variance of the elements for all the classes
        self.__var = np.zeros(len(classes),2)

        # Computing __p_y
        for i in indices:
            self.__p_y[i] += 1

        div = len(self.__p_y)

        for i in range(len(self.__p_y)):
            self.__p_y[i] /= div

        # Computing __moy

        #Attention à tester si attributes grandis bien en Y
        #normalement ca devrait aller chopper tout les éléments de X et les placer dans la colonne correspondante à la classe!! à tester!!!
        attributes = np.empty([len(classes)])
        for i in range(len(indices)):
            attributes[indices[i]].append(X[i])
            attributes[indices[i]].append(X[i])

        #on regarde par rapport à toutes les classes et on prend la moyenne des attibuts de chaque classe
        for i in attributes:
            self.__moy[i] = np.mean(attributes, axis=1) 
        #mtn ca devrait etre bon, normalement chaque élément de moy va correspondre à la moyenne des éléments de X en fonction de la classe!! à tester et vérifier!!!


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

        # ====================
        # TODO your code here.
        # ====================

        pass

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

        # ====================
        # TODO your code here.
        # ====================

        pass

if __name__ == "__main__":
    #from data import make_data
    #from plot import plot_boundary

    dataset_size = 2000
    trainingSet_size = 150
