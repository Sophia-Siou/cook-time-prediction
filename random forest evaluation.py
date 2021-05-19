''''***************************************************************
*    Proj 2 random forest
*    Author: Sophia Xiao (Student ID 1072038)
*    Date: May 1st, 2021
*    Dataset Source: Generating Personalized Recipes from Historical User Preferences.
*    Bodhisattwa Prasad Majumder,Shuyang Li, Jianmo Ni, Julian McAuley, in Proceedings of the 2019 Conference on Empirica
*    Methods in Natural Language Processing and the 9th International Joint Conference on Natural
*    Language Processing (EMNLP-IJCNLP), 2019.
*
*
****************************************************************/'''

import pandas as pd
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

df = pd.read_csv("recipe_train.csv")
df2 = pd.read_csv("recipe_test.csv")
stepdf = pd.read_csv("train_steps_doc2vec50.csv", header=None)
ingdf = pd.read_csv("train_ingr_doc2vec50.csv", header=None)
# get train col ['name', 'n_steps', 'n_ingredients', 'steps', 'ingredients',
#        'duration_label'],

steptest = pd.read_csv("test_steps_doc2vec50.csv", header=None)
ingtest = pd.read_csv("test_ingr_doc2vec50.csv", header=None)

# create a new dataframe containing numeric training data
data1 = df[['n_steps', 'n_ingredients']].copy()
data = pd.concat([data1, stepdf, ingdf], axis=1)

# test data
data2 = df2[['n_steps', 'n_ingredients']].copy()
test_data = pd.concat([data2, steptest, ingtest], axis=1)

# initialize variable name from train set
X = data[[i for i in data.columns]]
y = df['duration_label']

# features in test file
x_test = test_data[[i for i in test_data.columns]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

train_results = []
test_results = []
list_nb_trees = [10, 20, 50, 100]

for nb_trees in list_nb_trees:
    rf = RandomForestRegressor(n_estimators=nb_trees)
    rf.fit(X_train, y_train)

    train_results.append(mean_squared_error(y_train, rf.predict(X_train)))
    test_results.append(mean_squared_error(y_test, rf.predict(X_test)))

line1, = plt.plot(list_nb_trees, train_results, color="r", label="Training Score")
line2, = plt.plot(list_nb_trees, test_results, color="g", label="Testing Score")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('MSE')
plt.xlabel('n_estimators')
plt.savefig('tree vs error.png')
plt.show()