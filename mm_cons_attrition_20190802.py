# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:03:42 2019

@author: Gerard Mazi
@phone: 862.221.2477
@email: gerard.mazi@gmail.com
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Load consumer money market data
mm_raw = pd.read_csv('mm_in.csv')

# Stratified random sample by customer ID
mm = mm_raw.groupby('AccountNumber').apply(lambda x: x.sample(n=1)).reset_index(drop=True)

# Set target (CLOSED/INACTIVE/DORMANT = 1; OPEN = 0)
mm_int.loc[mm_int.AccountStatus == 'OPEN','AccountStatus'] = 0
mm_int.loc[mm_int.AccountStatus != 0,'AccountStatus'] = 1

# Set target variables
vars01 = ['AccountStatus','Seasoning','Spread','Balance']

# Subset based on target variables
mm = mm_int[vars01]

#...................................................................................................
# KNN
from sklearn.neighbors import KNeighborsClassifier

# Initial model
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)

# Optimize model
neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # set up knn classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)
    
    # fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    # compute accuracy on training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    # compute accuracy on testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    
# Plot optimization frontier
plt.title('knn: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# ROC curve and AUC
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_scores = knn.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:,1])
roc_auc = auc(fpr, tpr)

plt.title('ROC KNN')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#...................................................................................................
# LOGISTIC 
from sklearn.linear_model import LogisticRegression

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Logistic model
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X_train, y_train)
y_pred = logreg_cv.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))

# ROC Curve
y_pred_prob = logreg_cv.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.title('ROC LOGISTIC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#...................................................................................................
# DECISION TREE
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 3),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Tree Model
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X_train, y_train)
y_pred = tree_cv.predict(X_test)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC Curve and AUC
y_pred_prob = tree_cv.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.title('ROC TREE')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
