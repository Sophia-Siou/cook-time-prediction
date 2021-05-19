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
import matplotlib.pyplot as plt
df = pd.read_csv("zeroR.csv")
df2 = pd.read_csv("knn.csv")
df3 = pd.read_csv("tree.csv")


df['duration_label'].plot(kind='hist')
plt.savefig('zeroR output.png')
plt.show()

'''df2['duration_label'].plot(kind='hist')
plt.savefig('tree output.png')
plt.show()'''

'''df3['duration_label'].plot(kind='hist')
plt.savefig('knn output.png')
plt.show()'''