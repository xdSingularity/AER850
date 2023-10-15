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
X = df[['X','Y','Z']]
Y = df['Step']

from sklearn.model_selection import StratifiedShuffleSplit
# Assuming you're stratifying based on the 'Step' column
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, Y):
    train_X, test_X = X.iloc[train_index], X.iloc[test_index]
    train_Y, test_Y = Y.iloc[train_index], Y.iloc[test_index]

#model 1 RANDOM FOREST CLASSIFIER
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100, random_state=42)
RFC.fit(train_X, train_Y)
RFC_predictions = RFC.predict(test_X)
RFC_test_accuracy = accuracy_score(RFC_predictions, test_Y)
print("Random Forest Classifier test accuracy (before best hyper parameters) is: ", round(RFC_test_accuracy, 5))

#model 2 LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000) # Increasing max_iter for convergence
LR.fit(train_X, train_Y)
LR_predictions = LR.predict(test_X)
LR_test_accuracy = accuracy_score(LR_predictions, test_Y)
print("Logistic Regression test accuracy (before best hyper parameters) is: ", round(LR_test_accuracy, 5))

#model 3 SVM (SVC)
from sklearn.svm import SVC
SVM_clf = SVC()  # Using a linear kernel for simplicity. You can try other kernels like 'rbf'.
SVM_clf.fit(train_X, train_Y)
SVM_predictions = SVM_clf.predict(test_X)
SVM_test_accuracy = accuracy_score(SVM_predictions, test_Y)
print("Support Vector Machine Classifier test accuracy (before best hyper parameters) is: ", round(SVM_test_accuracy, 5))

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

# Training RFC with best hyperparameters
RFC_best = RandomForestClassifier(**RFC_params)
RFC_best.fit(train_X, train_Y)
RFC_best_predictions = RFC_best.predict(test_X)

# Training LR with best hyperparameters
LR_best = LogisticRegression(max_iter=10000, **LR_params)
LR_best.fit(train_X, train_Y)
LR_best_predictions = LR_best.predict(test_X)

# Training SVM with best hyperparameters
SVM_best = SVC(**SVM_params)
SVM_best.fit(train_X, train_Y)
SVM_best_predictions = SVM_best.predict(test_X)

from sklearn.metrics import f1_score, precision_score, accuracy_score

# Metrics Calculation
def compute_metrics(predictions, true_values):
    accuracy = accuracy_score(predictions, true_values)
    precision = precision_score(predictions, true_values, average='weighted')  # Considering multi-class problem
    f1 = f1_score(predictions, true_values, average='weighted')  # Considering multi-class problem
    return accuracy, precision, f1

# RFC metrics
RFC_accuracy, RFC_precision, RFC_f1 = compute_metrics(RFC_best_predictions, test_Y)
print(f"\nRandom Forest Classifier - Accuracy: {round(RFC_accuracy, 5)}, Precision: {round(RFC_precision, 5)}, F1 Score: {round(RFC_f1, 5)}")

# LR metrics
LR_accuracy, LR_precision, LR_f1 = compute_metrics(LR_best_predictions, test_Y)
print(f"\nLogistic Regression - Accuracy: {round(LR_accuracy, 5)}, Precision: {round(LR_precision, 5)}, F1 Score: {round(LR_f1, 5)}")

# SVM metrics
SVM_accuracy, SVM_precision, SVM_f1 = compute_metrics(SVM_best_predictions, test_Y)
print(f"\nSupport Vector Machine Classifier - Accuracy: {round(SVM_accuracy, 5)}, Precision: {round(SVM_precision, 5)}, F1 Score: {round(SVM_f1, 5)}")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y, LR_best_predictions)  # Replace with your best model's predictions
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=LR.classes_, yticklabels=LR.classes_)  # Replace RFC with your best model
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
















