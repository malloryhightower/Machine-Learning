# -*- coding: utf-8 -*-
"""
Last Edited Mon May 27, 2019

@author: Mallory Hightower (with code from Chris)
"""

#import necessary packages
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from itertools import product
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt


# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each
        # done! 
# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparameter settings
        # done! 
# 3. find some simple data
         # done! 
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings
         # done! 
# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
         # done!
# 6. Investigate grid search function
         # done!

# load the data set
wine = load_wine()

# matrix M is the data 
M = wine.data

# array L is the target variables (they y values) for the data set
L = wine.target

# folds used for cross validation
n_folds = 5

# view the cross validation object
kf = KFold(n_splits=n_folds)
print(kf)

# pack the arrays together into "data"
data = (M,L,n_folds)

# view the data
print(data)

# just to see what kf.split does
for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    print("k fold = ", ids)
    print("            train indexes", train_index)
    print("            test indexes", test_index)

# "run" function runs all the classifiers on the data
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # Establish the cross validation 
  ret = {} # classic explication of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): # We're interating through train and test indexes by using kf.split
                                                                      # from M and L.
                                                                      # We're simply splitting rows into train and test rows
                                                                      # for our five folds.    
    clf = a_clf(**clf_hyper) # unpack paramters into clf if they exist   # this gives all keyword arguments except 
                                                                            # for those corresponding to a formal parameter
                                                                            # in a dictionary.
                                                                                   
    clf.fit(M[train_index], L[train_index])   # First param, M when subset by "train_index", 
                                                 # includes training X's. 
                                                 # Second param, L when subset by "train_index",
                                                 # includes training Y.                               
    pred = clf.predict(M[test_index])         # Using M -our X's- subset by the test_indexes, 
                                                 # predict the Y's for the test rows.    
    ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret

# Declare empty clfs Accuracy Dict to populate in myHyperSetSearch     
clfsAccuracyDict = {}

# A dictionary where scores are kept by model and hyper parameter combinations
# this is necessary because otherwise the results are overwritten when "run" executes
def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) # Since we have a number of k-folds for each classifier...
                           # We want to prevent unique k1 values due to different "key" values
                           # when we actually have the same classifer and hyper parameter settings.
                           # So, we convert to a string                       
        #String formatting            
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')       
        # Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) # append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] # create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)            
        
# function that runs through the hyperparameter combinations and matches it with the clf name
def myHyperParamSearch(clfsList,clfDict):  
    for clf in clfsList:    
    # check if values in clfsList are in clfDict ... 
        clfString = str(clf)     
        for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters   
            if k1 in clfString:
                #allows you to do all the matching key and values
                k2,v2 = zip(*v1.items()) # explain zip (https://docs.python.org/3.3/library/functions.html#zip)
                for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
                    hyperParams = dict(zip(k2, values)) # create a dictionary from their values
                    results = run(clf, data, hyperParams) # pass the clf and dictionary of hyper param combinations to run; get results
                    populateClfAccuracyDict(results) # populate clfsAccuracyDict with results

# the classifier combinations you want to run stored in a list
clfsList = [RandomForestClassifier, LogisticRegression, KNeighborsClassifier] 

# dictionary of the classifiers with the different hyperparameters that you want to run
clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], "n_jobs": [1,2,3], "max_depth": [3,5,8]},                                      
'LogisticRegression': {"tol": [0.001,0.01,0.1], "penalty": ['l1','l2'], "solver": ['liblinear','saga']},
'KNeighborsClassifier': {"n_neighbors": np.arange(3, 8), "weights": ['uniform', 'distance'], "algorithm": ['ball_tree', 'kd_tree', 'brute']}}

# Run myHyperSetSearch and print the results
myHyperParamSearch(clfsList,clfDict)    
print(clfsAccuracyDict)

# FIRST GRAPH: JUST LOOKING AT ACCURACY RANGES
# create new dict by averaging and rounding the accuracy scores
import statistics
newdict={k:round(statistics.mean(v),2) for k,v in clfsAccuracyDict.items()}
print(newdict)

lists = sorted(newdict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.xticks([])
plt.title('Range in Accuracy Scores',fontsize=30)
plt.xlabel('Classifiers and Hyperparameter combinations',fontsize=20)
plt.ylabel('Accuracy',fontsize=25)
plt.savefig('AllClassifier_Results.png',bbox_inches = 'tight')
plt.show()

"""
# INITIAL PLOTTING RESULTS
# It looks like some classifiers perform consistently mediocre (about 60% accuracy),
    # some classifiers perform consistently very well with around 85%-95% accuracy,
    # and some classifiers perform poorly with accuracies below 50% (the equivalent of flipping a coin).
# There are also some much more varied accuracy results, but we want to narrow in on the accuracy 
    # distributions of each specific classifier. 
"""

# Examine Box and Whisker Plots for each classifier
# first get each classifier into its own dictionary for easier plotting
LR={ k:v for k,v in clfsAccuracyDict.items() if 'LogisticRegression' in k }
l=len(LR)
KN={ k:v for k,v in clfsAccuracyDict.items() if 'KNeighborsClassifier' in k }
k=len(KN)
RF={ k:v for k,v in clfsAccuracyDict.items() if 'RandomForestClassifier' in k }
r=len(RF)
check=[l,k,r]
s=sum(check)
print("The difference in the original dict lengh and the sum of the new dicts is = \n", s-len(clfsAccuracyDict))

# Plot the logistic regression accuracy results
labels, data = [*zip(*LR.items())]  # 'transpose' items to parallel key, value lists

plt.boxplot(data)
plt.xticks([])
plt.title("Box Plots for Logistic Regression Hyperparamters",fontsize=30)
plt.xlabel('Hyperparameter Combinations',fontsize=20)
plt.ylabel('Accuracy',fontsize=25)
plt.savefig('LogisticRegression_Results.png',bbox_inches = 'tight')
plt.show()

# Plot the KNeighbors Classifier accuracy results
labels, data = [*zip(*KN.items())]  # 'transpose' items to parallel key, value lists

plt.boxplot(data)
plt.xticks([])
plt.title("Box Plots for KNeighbors Hyperparamters",fontsize=30)
plt.xlabel('Hyperparameter Combinations',fontsize=20)
plt.ylabel('Accuracy',fontsize=25)
plt.savefig('KNeighbors_Results.png',bbox_inches = 'tight')
plt.show()

# Plot the Random Forest Classifier accuracy results
labels, data = [*zip(*RF.items())]  # 'transpose' items to parallel key, value lists

plt.boxplot(data)
plt.xticks([])
plt.title("Box Plots for Random Forest Hyperparamters",fontsize=30)
plt.xlabel('Hyperparameter Combinations',fontsize=20)
plt.ylabel('Accuracy',fontsize=25)
plt.savefig('RandomForest_Results.png',bbox_inches = 'tight')
plt.show()

"""
# DISCUSS BOX PLOT RESULTS
# From the box plots it looks like the best performing classifier is Random Forest. Not surprising. Random Forest 
    # is one of my favorite classifiers for its interpretability and performance.
# The random forest results are consistently between 85% and 95% accuracy with a tighter distribution and fewer 
    # outliers. 
# The logistic regression classifier has results all over the place with accuracy distributions between 0% 
    # and 100%. These results have the largest distribution and the highest variabilty.
# KNeighbors looks like it performs well but consistently has some very low minimum accuracy scores. Probably
    # as a result of not having enough Neighbors to make the correct classification. Otherwise, KNeighbors 
    # has a tight distribution and low variance with an average accuracy between 50% and 80%.
# Since Random Forest was the best performing, we will next determine which hyperparameter combination of RF
    # results in the highest accuracy. 
"""

# EXAMINE RANDOM FOREST HYPERPARAMETER COMBINATIONS
# Plot the Random Forest Classifier accuracy results
# As you can see, it does not really help but using the labels since they are so long!!!
# We will just have to eyeball it.
labels, data = [*zip(*RF.items())]  # 'transpose' items to parallel key, value lists

plt.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
plt.title("Box Plots for Random Forest Hyperparamters",fontsize=30)
plt.xlabel('Hyperparameter Combinations',fontsize=20)
plt.ylabel('Accuracy',fontsize=25)
plt.savefig('RandomForest_Hyperperameters.png',bbox_inches = 'tight')
plt.show()

"""
# EXAMINE HYPERPARAMETERS OF OPTIMAL CLASSIFIER
# Just from looking at the box plots, the 17th box plot seems to be one of the best, if not the best, 
    # performing hyperparameter combination. The box plot is a tight distribution 
    # with no significant outliers and an average accuracy of about 94%. 
# Due to the difficulty
    # in graphing with the super long labels, we are just eyeballing the graph, which is not the ideal method.
# These are the ideal hyperparameters combinations for the RF classifier on this data set.
    # n_jobs=3, max_depth=8, and min_samples_split=3
"""    
print(list(RF.keys())[17])

# INVESTIGATING GRID SEARCH IN PYTHON
from sklearn.model_selection import GridSearchCV
# Using the same data as other grid search example
wine = load_wine()
M = wine.data
L = wine.target

# different models for classifier comparison
models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'KNeighboursClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression()
}

# the hyperparameters you want to test for each classifier
# using the same combinations as above 
params = {
    'RandomForestClassifier':{ 
           "min_samples_split": [2,3,4], "n_jobs": [1,2,3], "max_depth": [3,5,8] 
            },
    'KNeighboursClassifier': {
            "n_neighbors": np.arange(3, 8), "weights": ['uniform', 'distance'], 
            "algorithm": ['ball_tree', 'kd_tree', 'brute']
        },
    'LogisticRegression': {
            "tol": [0.001,0.01,0.1], "penalty": ['l1','l2'], "solver": ['liblinear','saga']
        }  
}
    
# empty dictionaries for the function    
params2 = {'RandomForestClassifier':{},
           'KNeighboursClassifier':{},
           'LogisticRegression':{}}

# the grid seach CV funcion 
def fit(train_data, train_target):
        """
        fits the list of models to the training data, thereby obtaining in each 
        case an evaluation score after GridSearchCV cross-validation
        """
        for name in models.keys():
            est = models[name]
            est_params = params2[name]
            gscv = GridSearchCV(estimator=est, param_grid=est_params, cv=5)
            gscv.fit(train_data, train_target)
            print("best parameters are: {}".format(gscv.best_estimator_))
            print("Where we selected the parameters: {}" .format(gscv.cv_results_['params'][gscv.best_index_]))
            print("with mean cross-validated score: {}" .format(gscv.best_score_))
            
# Execute the grid search function
fit(M, L)

# Did grid search find the same hyperparameter combination we found for random forest using the manual method??
# the optimal hyperparameter combinations for RF are below
# a is the combination found with the manual grid search
# b is the combination found with the gridsearchCV 
from difflib import SequenceMatcher
a="RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)"
b="RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=8, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=3, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=3, oob_score=False, random_state=None, verbose=0, warm_start=False)"
ratio = SequenceMatcher(None, a, b).ratio()
print("The similarity ratio of these two hyperparameters combinations is ", round(ratio,2))

"""
# ANALYZING GRID SEARCH RESULTS
# The hyperperameter combinations that the automatic grid search produced are different with a similarity of 75%,
    # just used to quickly see if the strings were the same.
# The automatic grid search in python chose all different parameter combinations from the parameters that we 
    # were varying (max_depth, the min_samples_split, and the n_jobs). That is ok because we were just eyeballing 
    # the box plot results. There is not significant difference in the mean accuracy of the two hyperparameter combinations,
    # meaning that both produce similarly high accuracy scores.
# Overall, the automatic grid search was easier to execute and had fewer lines of code. It also makes it much easier
    # to quickly identify the optimal hyperparameter combinations instead of trying to extract all the information
    # out of the dictionaries in the manual method.
    
"""    