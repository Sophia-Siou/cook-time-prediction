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
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(x_test)
'''print(y_pred)
print(len(y_pred))'''

lines = [i for i in range(1, len(y_pred) + 1)]
print(lines)

# write the output to csv file
with open("tree.csv", "w", newline='') as file:
    headers = ["id", "duration_label"]
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    for i, j in zip(lines, y_pred):
        writer.writerow({'id': i, 'duration_label': j})

print("output file executed.")