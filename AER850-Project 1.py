# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:23:17 2023

@author: ilanb
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Project 1 Data.csv')

## part 2 ###################################################################################################################
# using numpy as stated by the project description although unnecessary
x = np.array(df['X'])
y = np.array(df['Y'])
z = np.array(df['Z'])

plot1 = plt.figure()
ax1 = plot1.add_subplot(111, projection='3d')

ax1.set_xlabel('X-axis Label')
ax1.set_ylabel('Y-axis Label')
ax1.set_zlabel('Z-axis Label')

ax1.scatter(x, y, z, label='Data Points', c='g', marker='o')
ax1.set_title('3D Scatter Plot')
ax1.legend()
plt.show()

## part 3 ###################################################################################################################

# 1. Compute the Correlation
correlation_matrix = df.iloc[:, :-1].corr()

# 2. Visualize the Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

## part 4 ###################################################################################################################
X = df['X','Y','Z']
Y = df['Step']

from sklearn.model_selection import StratifiedShuffleSplit
# Assuming you're stratifying based on the 'Z' column
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, Y):
    train_X, test_X = X.iloc[train_index], X.iloc[test_index]
    train_Y, test_Y = Y.iloc[train_index], Y.iloc[test_index]

#model 1 RANDOM FOREST CLASSIFIER
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100, random_state=42)
RFC.fit(train_X, train_Y)
RFC_predictions = RFC.predict(train_X)
RFC_train_accuracy = accuracy_score(RFC_predictions, train_Y)
print("Random Forest Classifier training accuracy is: ", round(RFC_train_accuracy, 5))

#model 2 LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000) # Increasing max_iter for convergence
LR.fit(train_X, train_Y)
LR_predictions = LR.predict(train_X)
LR_train_accuracy = accuracy_score(LR_predictions, train_Y)
print("Logistic Regression training accuracy is: ", round(LR_train_accuracy, 5))

#model 3 SVM (SVC)
from sklearn.svm import SVC
SVM_clf = SVC(kernel="linear")  # Using a linear kernel for simplicity. You can try other kernels like 'rbf'.
SVM_clf.fit(train_X, train_Y)
SVM_predictions = SVM_clf.predict(train_X)
SVM_train_accuracy = accuracy_score(SVM_predictions, train_Y)
print("Support Vector Machine Classifier training accuracy is: ", round(SVM_train_accuracy, 5))

# GRIDSEARCH CV
from sklearn.model_selection import GridSearchCV
models = [
    {
        'name': 'RandomForestClassifier',
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [10, 30, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    },
    {
        'name': 'LogisticRegression',
        'model': LogisticRegression(random_state=42, max_iter=10000),
        'param_grid': [{'penalty': ['l1'],'C': [0.001, 0.01, 0.1, 1, 10, 100],'solver': ['liblinear', 'saga']},
            {'penalty': ['l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100],'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
            {'penalty': [None],'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}
        ]
    },
    {
        'name': 'SVC',
        'model': SVC(random_state=42),
        'param_grid': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
    }
]

best_params_dict = {}
for model_info in models:
    print(f"\nOptimizing {model_info['name']}...")
    grid_search = GridSearchCV(model_info['model'], model_info['param_grid'], cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
    grid_search.fit(train_X, train_Y)
    best_params = grid_search.best_params_
    best_params_dict[model_info['name']] = best_params
    print(f"Best Hyperparameters for {model_info['name']}:", best_params)

RFC_params = best_params_dict['RandomForestClassifier']
LR_params = best_params_dict['LogisticRegression']
SVM_params = best_params_dict['SVC']

## part 5 ###################################################################################################################


















