# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:15:26 2023

@author: ilanb
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")

print(df.info())

from sklearn.model_selection import train_test_split
train_set, train_set = train_test_split(df, test_size=0.2, random_state=75475487)


df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
print(df["income_cat"].value_counts())

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
for a in (strat_train_set,strat_test_set):
    a.drop("income_cat", axis=1)

#creating a scatter plot using panda
#from pandas.plotting import scatter_matrix
#attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
#scatter_matrix(df[attributes], figsize=(12, 8))

#looking for correlations
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix));
#corr_matrix["median_house_value"].sort_values(ascending=False)
#plt.matshow(corr_matrix)
selectedVariables = ['longitude', 'housing_median_age', 'total_rooms', 'median_income', 'ocean_proximity']

strat_train_set_selected = strat_train_set[selectedVariables]

















