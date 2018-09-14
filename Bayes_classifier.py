#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:20:07 2018

@author: manuel
"""

from __future__ import print_function, division
from builtins import range, input

import util
import numpy as np
import matlib.pyplot as plt
#needed below for drawing samples
from scipy.stats import multivariate_normal as mvn


class BayesClassifier:
    def fit(self, X,Y):
        # We assume here that the classes are numbered 0... K-1
        self.K = len(set(Y))
        
        self.gaussians = []
        for K in range(self.K):
            Xk = X[Y == k]
            mean = Xk.mean(axis =0)
            covar = np.cov(Xk.T)
            g = {'m':mean, 'c':covar}
            self.gaussians.append(g)
            #Creates a dictionary of guassians with mean 'm' and covariance 'c'
            
            
            
            #function for drawing a sample from a given class y
            def sample_given_y(self,y):
                g = self.gaussians[y]
                return mvn.rvs(mean = g['m'], covar = g['c'])
            #function for grabbing a sample for any class
            def sample(self):
                y = np.random.randint(self.K)
                return self.sample_given_y(y)
            
            if __name__ == '__main__':
                #gather MNIST data
                X,Y = util.get_mnist()
                #create an instance of the Bayes Classifier
                clf = BayesClassifier()
                #fit the Classifier to our data, in this case MNIST
                clf.fit(X,Y)
                
                for k in range(clf,K):
                    #show one sample and the mean image learned
                    
                    sample = clf.sample_given_y(k).reshape(28,28)
                    mean = clf.gaussians[k]['m'].reshape(28,28)
                    
                    plt.subplot(1,2,1)
                    plt.imshow(sample,cmap='gray')
                    plt.title('Sample')
                    plt.subplot(1,2,2)
                    plt.imshow(mean, cmap='gray')
                    plt.title('Mean')
                    plt.show()
                    
                    #generate a random sample
                    sample = clf.sample().reshape(28,28)
                    plt.imshow(sample,cmap='gray')
                    plt.title('Random sample from a Random Class')
                    
                
                
                