# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:21:19 2023

@author: ilanb
"""

import numpy as np 
import pandas as pd
#from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
df = pd.read_csv("housing.csv")

print(df.isna().any(axis=0).sum()) #how many columns have missing values
print(df.isna().any(axis=1).sum()) #how many rows have missing values
df = df.dropna()
df = df.reset_index()

from sklearn.preprocessing import OneHotEncoder
my_encoder = OneHotEncoder(sparse_output=False)
my_encoder.fit(df[['ocean_proximity']])
encoded_data = my_encoder.transform(df[['ocean_proximity']])
category_names = my_encoder.get_feature_names_out()
encoded_data_df = pd.DataFrame(encoded_data, columns = category_names)
df = pd.concat([df, encoded_data_df], axis = 1)
df = df.drop(columns = 'ocean_proximity')


#stratified sampling 
df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
for a in (strat_train_set,strat_test_set):
    a.drop("income_cat", axis=1)
    
# anything from here, unless for testing, should be only from the train dataset
#to avoid data snooping bias

train_y = strat_train_set['median_house_value']
train_x = strat_train_set.drop(columns = ["median_house_value", "income_cat"])

from sklearn.preprocessing import StandardScaler
my_scaler = StandardScaler()
my_scaler.fit(train_x.iloc[:,0:-5])
scaled_data = my_scaler.transform(train_x)
scaled_data_df = pd.DataFrame(scaled_data, columns=train_x.columns)

scaled_data_df.to_csv("mydataset.csv")

#creating scatter plot using panda
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
pd.plotting.scatter_matrix(df[attributes], figsize=(12, 8))

#looking at correlations
corr_matrix = strat_train_set.corr(numeric_only=True)
plt.figure()






















































