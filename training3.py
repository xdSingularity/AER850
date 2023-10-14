# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:37:14 2023

@author: ilanb
"""

import pandas as pd 
csv_file = 'data.csv'
df = pd.read_csv(csv_file)

# Displaying the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Getting basic information about the DataFrame
print("\nDataFrame info:")
print(df.info())

# Accessing columns and rows
print("\nSelecting a specific column:")
column_data = df['Column_Name']
print(column_data)

print("\nSelecting a specific row:")
row_data = df.loc[2]
print(row_data)