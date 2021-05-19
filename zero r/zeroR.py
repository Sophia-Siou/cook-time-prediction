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

df = pd.read_csv("recipe_train.csv")
df2 = pd.read_csv("recipe_test.csv")

# find the most freq label in class
freq = df['duration_label'].mode().values[0]
print(freq)


'''# output the prediction value
with open("zeroR.csv", "w", newline='') as file:
    headers = ["id", "duration_label"]
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    for i in range(1, len(df2) + 1):
        writer.writerow({'id': i, 'duration_label': freq})

print("output file executed.")'''

