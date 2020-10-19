#######################################################################################################################
#######################################################################################################################

###################################################################
# Funny Money B: Machine Training                                 #
# Created by Aniket N Prabhu                                      #
# References: pml51.py, pml62.py, pml73.py, pml75a.py, pml74b.py, #
#             pml88.py, pml91.py, pml94.py, pml136.py             #
###################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

moneydf = pd.read_csv('data_banknote_authentication.txt')
moneydf.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
X = moneydf.iloc[:, 0:4].values
Y = moneydf.iloc[:, 4].values
# print(X)
# print(Y)

# Splitting data 70% for training and 30% for testing
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)
SS = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
SS.fit(Xtrain)  # Fit calculates how to change the data based on what you're doing
stdXtrain = SS.transform(Xtrain)
stdXtest = SS.transform(Xtest)  # Mean and Std Dev not required for test sets

#######################################################################################################################

# Method 1: Perceptron ################################################################################################
print("# Perceptron ##################################################################################################")
Perc = Perceptron(max_iter=7, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)  # 7

# This will train the perceptron
Perc.fit(stdXtrain, Ytrain)

# Now, using the training on the test set
PredY = Perc.predict(stdXtest)
print('\nMisclassified samples: %d' % (Ytest != PredY).sum())
print('Accuracy: %.2f' % accuracy_score(Ytest, PredY))
stdXcomb = np.vstack((stdXtrain, stdXtest))  # Vertically stacking training and test sets for comparison
Ycomb = np.hstack((Ytrain, Ytest))  # Horizontally stacking Y sets
print('Number in combined ', len(Ycomb))
predYcomb = Perc.predict(stdXcomb)
print('Misclassified combined samples: %d' % (Ycomb != predYcomb).sum())
print('Combined Accuracy: %.2f' % accuracy_score(Ycomb, predYcomb))  # Accuracy classification score

#######################################################################################################################

# Method 2: Logistic Regression #######################################################################################
print("\n# Logistic Regression #######################################################################################")
LogReg = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=0)
# 'liblinear' is a library. 'ovr' means One-vs-rest
LogReg.fit(stdXtrain, Ytrain)

# Now, using the training on the test set
PredY = LogReg.predict(stdXtest)
print('\nMisclassified samples: %d' % (Ytest != PredY).sum())
print('Accuracy: %.2f' % accuracy_score(Ytest, PredY))
stdXcomb = np.vstack((stdXtrain, stdXtest))  # Vertically stacking training and test sets for comparison
Ycomb = np.hstack((Ytrain, Ytest))  # Horizontally stacking Y sets
print('Number in combined ', len(Ycomb))
predYcomb = LogReg.predict(stdXcomb)
print('Misclassified combined samples: %d' % (Ycomb != predYcomb).sum())
print('Combined Accuracy: %.2f' % accuracy_score(Ycomb, predYcomb))  # Accuracy classification score

#######################################################################################################################

# Method 3: Support Vector Machine ####################################################################################
print("\n# Linear SVM ###############################################################################################")
svm = SVC(kernel='linear', random_state=0, C=1.0)
svm.fit(stdXtrain, Ytrain)

# Now, using the training on the test set
PredY = svm.predict(stdXtest)
print('\nMisclassified samples: %d' % (Ytest != PredY).sum())
print('Accuracy: %.2f' % accuracy_score(Ytest, PredY))
stdXcomb = np.vstack((stdXtrain, stdXtest))  # Vertically stacking training and test sets for comparison
Ycomb = np.hstack((Ytrain, Ytest))  # Horizontally stacking Y sets
print('Number in combined ', len(Ycomb))
predYcomb = svm.predict(stdXcomb)
print('Misclassified combined samples: %d' % (Ycomb != predYcomb).sum())
print('Combined Accuracy: %.2f' % accuracy_score(Ycomb, predYcomb))  # Accuracy classification score

#######################################################################################################################

# Method 4: Decision Tree Learning ####################################################################################
print("\n# Decision Tree Learning ####################################################################################")
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
tree.fit(stdXtrain, Ytrain)

# Now, using the training on the test set
PredY = tree.predict(stdXtest)
print('\nMisclassified samples: %d' % (Ytest != PredY).sum())
print('Accuracy: %.2f' % accuracy_score(Ytest, PredY))
stdXcomb = np.vstack((stdXtrain, stdXtest))  # Vertically stacking training and test sets for comparison
Ycomb = np.hstack((Ytrain, Ytest))  # Horizontally stacking Y sets
print('Number in combined ', len(Ycomb))
predYcomb = tree.predict(stdXcomb)
print('Misclassified combined samples: %d' % (Ycomb != predYcomb).sum())
print('Combined Accuracy: %.2f' % accuracy_score(Ycomb, predYcomb))  # Accuracy classification score

#######################################################################################################################

# Method 5: Random Forest #############################################################################################
print("\n# Random Forest #############################################################################################")
forest = RandomForestClassifier(criterion='entropy', n_estimators=15, random_state=1, n_jobs=2)  # 10 15
forest.fit(stdXtrain, Ytrain)

# Now, using the training on the test set
PredY = forest.predict(stdXtest)
print('\nMisclassified samples: %d' % (Ytest != PredY).sum())
print('Accuracy: %.2f' % accuracy_score(Ytest, PredY))
stdXcomb = np.vstack((stdXtrain, stdXtest))  # Vertically stacking training and test sets for comparison
Ycomb = np.hstack((Ytrain, Ytest))  # Horizontally stacking Y sets
print('Number in combined ', len(Ycomb))
predYcomb = forest.predict(stdXcomb)
print('Misclassified combined samples: %d' % (Ycomb != predYcomb).sum())
print('Combined Accuracy: %.2f' % accuracy_score(Ycomb, predYcomb))  # Accuracy classification score

#######################################################################################################################

# Method 6: K-Nearest Neighbor ########################################################################################
print("\n# K-Nearest Neighbor ########################################################################################")
knn = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')  # 16 14 12 2
knn.fit(stdXtrain, Ytrain)

# Now, using the training on the test set
PredY = knn.predict(stdXtest)
print('\nMisclassified samples: %d' % (Ytest != PredY).sum())
print('Accuracy: %.2f' % accuracy_score(Ytest, PredY))
stdXcomb = np.vstack((stdXtrain, stdXtest))  # Vertically stacking training and test sets for comparison
Ycomb = np.hstack((Ytrain, Ytest))  # Horizontally stacking Y sets
print('Number in combined ', len(Ycomb))
predYcomb = knn.predict(stdXcomb)
print('Misclassified combined samples: %d' % (Ycomb != predYcomb).sum())
print('Combined Accuracy: %.2f' % accuracy_score(Ycomb, predYcomb))  # Accuracy classification score

#######################################################################################################################
# End #################################################################################################################
#######################################################################################################################
