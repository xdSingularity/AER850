# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 00:49:56 2023

@author: ilanb
"""


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Project 1 Data.csv')


## part 4 ###################################################################################################################
X = df[['X','Y','Z']]
Y = df['Step']

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

from sklearn.metrics import accuracy_score
#model 3 SVM (SVC)
from sklearn.svm import SVC
SVM = SVC()  # Using a linear kernel for simplicity. You can try other kernels like 'rbf'.
SVM.fit(train_X, train_Y)
SVM_predictions = SVM.predict(test_X)
SVM_test_accuracy = accuracy_score(SVM_predictions, test_Y)
print("Support Vector Machine Classifier test accuracy (before best hyperparameters) is: ", round(SVM_test_accuracy, 5))


# GRIDSEARCH CV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

SVM = SVC(random_state=42)
print("\nOptimizing SVC....")
grid_search = GridSearchCV(SVM, param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
grid_search.fit(train_X, train_Y)
best_params = grid_search.best_params_
print("\nBest Hyperparameters for SVC:", best_params)


## part 5 ###################################################################################################################

# Training SVM with best hyperparameters
C_best = grid_search.best_params_['C']
gamma_best = grid_search.best_params_['gamma']
kernel_best = grid_search.best_params_['kernel']
SVM_best = SVC(C=C_best, gamma=gamma_best, kernel=kernel_best, random_state=42)
#SVM_best = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'], random_state=42)
SVM_best.fit(train_X, train_Y)
SVM_best_predictions = SVM_best.predict(test_X)
SVM_best_test_accuracy = accuracy_score(test_Y, SVM_best_predictions)
print("Support Vector Machine Classifier test accuracy is: ", round(SVM_best_test_accuracy, 5))


















