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
        for i in range(len(indices))
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
        
        y = []
        for h in range(len(X)):             # Number of loops = number of samples
            max = 0
            for i in range(len(self.__classes)):
                Py = self.__p_y[i]          # Product of the different probabilities for one class
                for j in range(len(X[0]):   # Number of loops = number of features
                    exp_num = math.pow(X[h][j]-self.__moy[i][j], 2)
                    exp_den = 2*self.__var[i][j]
                    exp = math.exp(-exp_num/exp_den)
                    factor_den = math.pow(2*math.pi*self.__var[i][j], 1/2)
                    Py *= (1/factor_den)*exp
                if Py > max:
                   predict_classe = self.__classes[i]
                   max = Py
            y[h] = predict_classe

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

        # ====================
        # TODO your code here.
        # ====================

        pass

if __name__ == "__main__":
    #from data import make_data
    #from plot import plot_boundary

    dataset_size = 2000
    trainingSet_size = 150
