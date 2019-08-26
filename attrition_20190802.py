# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 2019

@author: Gerard Mazi
@phone: 862.221.2477
@email: gerard.mazi@gmail.com
"""
# Default Models under various ML methodologies

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import scale


# Import dataset
loan_raw = pd.read_csv("loan.csv")

# Select key variables
var_1 = [
    'loan_status',
         'loan_amnt','installment','sub_grade','emp_length','home_ownership','annual_inc','revol_util',
         'verification_status','purpose','dti','earliest_cr_line','inq_last_6mths','revol_bal'
]
loan_int = loan_raw[var_1]

# Retain fully paid and charged off
loan_int = loan_int[loan_int.loan_status.isin(['Fully Paid', 'Charged Off'])]

# Define target
loan_int['loan_status'] = np.where(loan_int.loan_status == 'Fully Paid', 0, 1)

# Select subset of key variables
var_2 = [
    'loan_status',
         'loan_amnt','installment','sub_grade','home_ownership','annual_inc','revol_util',
         'verification_status','purpose','dti','inq_last_6mths','revol_bal'
]
mm = loan_int[var_2]

# Clean NA
mm = mm.dropna(axis=0, how='any')

# Parse out dummy variables
mm = pd.get_dummies(mm, drop_first=True)

#########################################################################################################
# KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier

# Sample prep
y = mm[['loan_status']].values
X = mm[['loan_amnt','annual_inc','dti','inq_last_6mths','revol_bal','installment','revol_util']].values
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

# Base model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train.ravel())
knn.score(X_test, y_test)
y_pred_prob = knn.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred_prob)

# Optimize K
neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train.ravel())

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Plot optimization frontier
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Run optimal model
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train.ravel())

# ROC and AUC
y_pred_prob = knn.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('ROC CURVE')
plt.plot(fpr, tpr, 'b', label='KNN AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()


#########################################################################################################
# LOGISTIC
from sklearn.linear_model import LogisticRegression

# Sample prep
y = mm[['loan_status']].values
X = mm.drop('loan_status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

# Base model
logreg = LogisticRegression()
logreg.fit(X_train, y_train.ravel())
logreg.score(X_test, y_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)

# Optimal model
# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Logistic model
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc')
logreg_cv.fit(X_train, y_train.ravel())
logreg_cv.best_score_
logreg_cv.best_params_

# ROC Curve
y_pred_prob = logreg_cv.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.plot(fpr, tpr, 'b', label = 'LOGREG AUC = %0.2f' % roc_auc, color='green')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

################################################################################################################
# CLASSIFICATION TREE
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier

# Sample prep
y = mm[['loan_status']].values
X = mm.drop('loan_status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

# Base model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_prob = tree.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred_prob)

# Optimized model
# Setup the parameters and distributions to sample from: param_dist
param_dist = {
    "max_depth": [3, None],
    "max_features": randint(1, 11),
    "min_samples_leaf": randint(1, 9),
    "criterion": ["gini", "entropy"]
}

# Tree Model
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, scoring='roc_auc', cv=5)
tree_cv.fit(X_train, y_train)
tree_cv.best_score_
tree_cv.best_params_

# ROC Curve and AUC
y_pred_prob = tree_cv.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label = 'TREE AUC = %0.2f' % roc_auc, color='orange')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

################################################################################################################
# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

# Sample prep
y = mm[['loan_status']].values
X = mm.drop('loan_status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

# Base model
rf = RandomForestClassifier(random_state=666)
rf.fit(X_train, y_train.ravel())
y_pred_prob = rf.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred_prob)

# Optimized model
param_grid = {
    'n_estimators': [40, 60, 80, 100],
    'max_features': ['auto', 'log2'],
    'max_depth': [2, 6, 8, 10, 12],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [12, 18, 24]
}

rf = RandomForestClassifier(random_state=666, class_weight={0:1, 1:12}, bootstrap=True)
rf_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc')
rf_cv.fit(X_train, y_train.ravel())
rf_cv.best_score_
rf_cv.best_params_

# ROC Curve and AUC
y_pred_prob = rf_cv.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label = 'RANDFOR AUC = %0.2f' % roc_auc, color='purple')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
