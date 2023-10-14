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
from sklearn.model_selection import StratifiedShuffleSplit
# Assuming you're stratifying based on the 'Z' column
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['Step']):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)

train_X = strat_train_set[['X', 'Y', 'Z']]
train_Y = strat_train_set['Step']


## part 5 ###################################################################################################################


















