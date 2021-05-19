================= Term of Use =================

The data has been collected from Food.com (formerly GeniusKitchen), under the provision that any resulting work should cite this resource:

Generating Personalized Recipes from Historical User Preferences. Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, Julian McAuley, in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019.


================= Files =================

1. recipe_train.csv
This file contains the recipe features and label for training instances.
Number of instances: 40000
Number of columns: 6
The columns are (the column names are in the first row):
	name, n_steps, n_ingredients, steps, ingredients, duration_label

The columns name, steps and ingredients contain the raw text data of these features.

The class label is in the last column: duration_label. There are 3 possible levels, 1, 2 or 3, which correspond to quick, medium and slow.

2. recipe_test.csv
This file contains the recipe features for test instances.
Number of instances: 10000
Number of columns: 5
The columns are (the column names are in the first row):
	name, n_steps, n_ingredients, steps, ingredients


3. recipe_text_features_*.zip: preprocessed text features for training and test sets, 1 zipped file for each text encoding method

3.1 recipe_text_features_countvec.zip
9 files
(1) train_name_countvectorizer.pkl
This file contains the CountVectorizer extracted using the text of the recipe "name" in the training set.
To load the file in Python:
	vocab = pickle.load(open("train_name_countvectorizer.pkl", "rb"))
	
To access the list of vocabulary (this will give you a dict):
	vocab_dict = vocab.vocabulary_
	
More about how to use the CountVectorizer can be found: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

(2) train_steps_countvectorizer.pkl
This file contains the CountVectorizer extracted using the text of the recipe "steps" in the training set.

(3) train_ingr_countvectorizer.pkl
This file contains the CountVectorizer extracted using the text of the recipe "ingredients" in the training set.

(4) train_name_vec.npz
This file contains a sparse matrix of the Bag-of-Word representation of the recipe names for training data.
The dense version of this matrix should be [40000 * size of vocabulary], and the element (i,j) in the matrix is the count of each vocabulary term j in instance i. The vocabulary corresponds to the vocabulary_ attribute of vocab (which can be checked as detailed in (1))

As a lot of elements in this matrix are zeros, it has been compressed to a sparse matrix. After loading, the sparse matrix can be used as a normal matrix for training or testing.

To load the sparse matrix:
	import scipy
	scipy.sparse.load_npz('train_name_vec.npz')

train_steps_vec.npz and train_ingr_vec.npz are the sparse matrices for recipe steps and ingredients, respectively.

(5) test_name_vec.npz
This file contains a sparse matrix of the Bag-of-Word representation of the recipe names for test data. 
The dense version of this matrix should be [10000 * size of vocabulary]. The vocabulary is the one that has been extracted from training, but the elements in this matrix are the counts for each recipe in the test set.

To load the sparse matrix:
	import scipy
	scipy.sparse.load_npz('test_name_vec.npz')
	
test_steps_vec.npz and test_ingr_vec.npz are the sparse matrices for recipe steps and ingredients, respectively.

3.2 recipe_text_features_doc2vec50.zip
6 files
(1) train_name_doc2vec50.csv
This file contains a matrix of Doc2Vec representation of the recipe names for training data, with 50 features.
The dimension of this matrix is [40000 * 50], and the element (i,j) in the matrix is a numeric value for feature j of an instance i. 

To load the matrix:
	import pandas as pd
	pd.read_csv(r"train_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)

train_steps_doc2vec50.csv and train_ingr_doc2vec50.csv are the matrices for recipe steps and ingredients, respectively.

(2) test_name_doc2vec50.csv
This file contains a matrix of Doc2Vec representation of the recipe names for test data, with 50 features extracted from the training data.
The dimension of this matrix is [10000 * 50], and the element (i,j) in the matrix is a numeric value for feature j of an instance i. 

To load the matrix:
	import pandas as pd
	pd.read_csv(r"test_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)

test_steps_doc2vec50.csv and test_ingr_doc2vec50.csv are the matrices for recipe steps and ingredients, respectively.

3.3 recipe_text_features_doc2vec100.zip
Similar to recipe_text_features_doc2vec50, except that 100 features are used for each instance