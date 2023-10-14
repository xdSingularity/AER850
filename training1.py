# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:40:31 2023

@author: ilanb
"""

fruits = ["apple", "banana", "orange"]

#looping through a list
print("looping through a list:")
for X in fruits:
    print(X)

# looping through a string
print("\nlooping through a string:")
for char in "Hello":
    print(char)
    
# looping through a range of numbers 
print("\nLooping through a range of numbers:")
for num in range(1, 5):
    print(num)
    
# looping through a dictionnary
print("\nLooping through a dictionnary:")
student = {"name": "John", "age": 20, "grade": "A"}
for key, value in student.items():
    print(key, ":", value)