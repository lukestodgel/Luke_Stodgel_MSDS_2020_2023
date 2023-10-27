# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:55:06 2023

@author: Luke
"""

# Homework 2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

# matplot lib accuracy plot code
def plot_accuracy(ret):
    models = [i[0] for i in ret]
    accuracy = [i[3]['accuracy'] for i in ret]
    plt.bar(models, accuracy)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of each model')
    plt.savefig('.//accuracy.png')
    plt.show()

# matplot lib auc plot code
def plot_auc(ret):
    models = [i[0] for i in ret]
    auc = [i[4]['AUC'] for i in ret]
    plt.bar(models, auc)
    plt.xlabel('Models')
    plt.ylabel('AUC')
    plt.title('AUC of each model')
    plt.savefig('.//AUC.png')
    plt.show()

# matplot lib f1_score plot code
def plot_f1_score(ret):
    models = [i[0] for i in ret]
    f1_score = [i[2]['best_f1_score'] for i in ret]
    plt.bar(models, f1_score)
    plt.xlabel('Models')
    plt.ylabel('F1 Score')
    plt.title('F1 Score of each model')
    plt.savefig('.//f1_score.png')
    plt.show()
    
# This function takes a clf_dictionary, data, and a clf_hyperparameter
# dictionary as inputs. It will create a train/test split using the input
# data and then for each classifier, it will cycle through every combination 
# of hyperparameters and return the best params based on f1_score. 
# The accuracy, f1_score and auc will be outputted next to the name of each
# classifier. The metrics for each model is saved in a list and used
# for plotting.
# This function will test logistic regression, random forest, KNN, SVM, 
# decision tree classifier, xgb, gaussian NB and multinomial NB models.
def custom_gridsearch(clf_dict, data, clf_hyper):
    
    #create X and Y
    X, y = data.data, data.target
    #create train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    for clf in clf_dict:
        if clf == 'lr':
            best_f1_score_lr = 0
            for C in clf_hyper['Logistic Regression']['C']:
                for solver in clf_hyper['Logistic Regression']['solver']:
                    for penalty in clf_hyper['Logistic Regression']['penalty']:
                        model = LogisticRegression(C=C, solver=solver,
                                                   penalty=penalty)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        f1_score_ = f1_score(y_test, y_pred)
                        auc_ = roc_auc_score(y_test, y_pred)
                        if f1_score_ > best_f1_score_lr:
                            best_params_lr = {'C': C, 'solver': solver, 
                                              'penalty': penalty}
                            best_f1_score_lr = f1_score_
                            corresponding_auc_ = auc_
                            corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_lr}, 
                        {'best_f1_score': round(best_f1_score_lr, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)},
                        {'AUC': round(corresponding_auc_, 3)}])
        elif clf == 'rf':
            best_f1_score_rf = 0
            for n_estimators in clf_hyper['RandomForestClassifier']['n_estimators']:
                for criterion in clf_hyper['RandomForestClassifier']['criterion']:
                    for max_depth in clf_hyper['RandomForestClassifier']['max_depth']:
                        model = RandomForestClassifier(n_estimators=n_estimators, 
                                                       criterion=criterion,
                                                       max_depth=max_depth)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        f1_score_ = f1_score(y_test, y_pred)
                        auc_ = roc_auc_score(y_test, y_pred)
                        if f1_score_ > best_f1_score_rf:
                            best_params_rf = {'n_estimators': n_estimators,
                                              'criterion': criterion,
                                              'max_depth': max_depth}
                            best_f1_score_rf = f1_score_
                            corresponding_auc_ = auc_
                            corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_rf}, 
                        {'best_f1_score': round(best_f1_score_rf, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)},
                        {'AUC': round(corresponding_auc_, 3)}])
        elif clf == 'knn':
            best_f1_score_knn = 0
            for n_neighbors in clf_hyper['KNeighborsClassifier']['n_neighbors']:
                for weights in clf_hyper['KNeighborsClassifier']['weights']:
                    for p in clf_hyper['KNeighborsClassifier']['p']:
                        model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                     weights=weights, p=p)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        f1_score_ = f1_score(y_test, y_pred)
                        auc_ = roc_auc_score(y_test, y_pred)
                        if f1_score_ > best_f1_score_knn:
                            best_params_knn = {'n_neighbors': n_neighbors,
                                               'weights': weights,
                                               'p': p}
                            best_f1_score_knn = f1_score_
                            corresponding_auc_ = auc_
                            corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_knn}, 
                        {'best_f1_score': round(best_f1_score_knn, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)},
                        {'AUC': round(corresponding_auc_, 3)}])
        elif clf == 'SVM':
            best_f1_score_svm = 0
            for C in clf_hyper['SVC']['C']:
                for kernel in clf_hyper['SVC']['kernel']:
                    model = SVC(C=C, kernel=kernel)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_score_ = f1_score(y_test, y_pred)
                    auc_ = roc_auc_score(y_test, y_pred)
                    if f1_score_ > best_f1_score_svm:
                        best_params_svm = {'C': C, 'kernel': kernel}
                        best_f1_score_svm = f1_score_
                        corresponding_auc_ = auc_
                        corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_svm}, 
                        {'best_f1_score': round(best_f1_score_svm, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)}, 
                        {'AUC': round(corresponding_auc_, 3)}])
        elif clf == 'dt':
            best_f1_score_dt = 0
            for criterion in clf_hyper['DecisionTreeClassifier']['criterion']:
                for max_depth in clf_hyper['DecisionTreeClassifier']['max_depth']:
                    model = DecisionTreeClassifier(criterion=criterion,
                                                   max_depth=max_depth)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_score_ = f1_score(y_test, y_pred)
                    auc_ = roc_auc_score(y_test, y_pred)
                    if f1_score_ > best_f1_score_dt:
                        best_params_dt = {'criterion': criterion, 
                                          'max_depth': max_depth}
                        best_f1_score_dt = f1_score_   
                        corresponding_auc_ = auc_
                        corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_dt}, 
                        {'best_f1_score': round(best_f1_score_dt, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)}, 
                        {'AUC': round(corresponding_auc_, 3)}])
        elif clf == 'xgb':
            best_f1_score_xgb = 0
            for max_depth in clf_hyper['XGBClassifier']['max_depth']:
                for n_estimators in clf_hyper['XGBClassifier']['n_estimators']:
                    model = xgb.XGBClassifier(max_depth=max_depth,
                                              n_estimators=n_estimators)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_score_ = f1_score(y_test, y_pred)
                    auc_ = roc_auc_score(y_test, y_pred)
                    if f1_score_ > best_f1_score_xgb:
                        best_params_xgb = {'max_depth': max_depth, 
                                           'n_estimators': n_estimators}
                        best_f1_score_xgb = f1_score_    
                        corresponding_auc_ = auc_
                        corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_xgb}, 
                        {'best_f1_score': round(best_f1_score_xgb, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)}, 
                        {'AUC': round(corresponding_auc_, 3)}])
        elif clf == 'gNB':
            best_f1_score_gNB = 0
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_score_ = f1_score(y_test, y_pred)
            auc_ = roc_auc_score(y_test, y_pred)
            if f1_score_ > best_f1_score_gNB:
                best_params_gNB = {'None'}
                best_f1_score_gNB = f1_score_    
                corresponding_auc_ = auc_
                corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_gNB}, 
                        {'best_f1_score': round(best_f1_score_gNB, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)}, 
                        {'AUC': round(corresponding_auc_, 3)}])
        elif clf == 'mNB':
            best_f1_score_mNB = 0
            for alpha in clf_hyper['MultinomialNB']['alpha']:
                for fit_prior in clf_hyper['MultinomialNB']['fit_prior']:
                    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_score_ = f1_score(y_test, y_pred)
                    auc_ = roc_auc_score(y_test, y_pred)
                    if f1_score_ > best_f1_score_mNB:
                        best_params_mNB = {'alpha': alpha,
                                           'fit_prior': fit_prior}
                        best_f1_score_mNB = f1_score_     
                        corresponding_auc_ = auc_
                        corresponding_accuracy = accuracy
            ret.append([clf, {'best_params': best_params_mNB}, 
                        {'best_f1_score': round(best_f1_score_mNB, 3)}, 
                        {'accuracy': round(corresponding_accuracy, 3)}, 
                        {'AUC': round(corresponding_auc_, 3)}])

# main function
if __name__ == '__main__':
    
    # load data
    data = load_breast_cancer()
    
    # create list of model tuples
    clfs = [('lr', LogisticRegression()),('rf', RandomForestClassifier()),
            ('knn', KNeighborsClassifier()),('SVM', SVC()),
            ('dt', DecisionTreeClassifier()),('xgb', xgb.XGBClassifier()),
            ('gNB', GaussianNB()), ('mNB', MultinomialNB())]
    
    # convert to dictionary
    clfs = dict(clfs)

    # create param_grid for each classifier
    param_grid = {'Logistic Regression': {'C': [0.1, 1, 10], 
                                          'solver': ['liblinear'], 
                                          'penalty': ['l1', 'l2']},
                  'KNeighborsClassifier': {'n_neighbors': [1, 3, 5, 10, 15, 20],
                                           'weights': ['uniform', 
                                                       'distance'],
                                           'p': [1, 2]},
                  'SVC': {'C': [0.1, 1, 10], 
                          'kernel': ['linear', 'poly', 'rbf']},
                  'DecisionTreeClassifier': {'criterion': ['gini', 'entropy'],
                                             'max_depth': [1, 3, 5, 10, 15, 20]},
                  'RandomForestClassifier': {'n_estimators': [5, 10, 100, 250],
                                             'criterion': ['gini', 'entropy'],
                                             'max_depth': [1, 3, 5, 10, 15, 20]},
                  'XGBClassifier': {'max_depth': [1, 3, 5, 10], 
                                    'n_estimators': [10, 100, 250, 500]},
                  'GaussianNB': {}, 
                  'MultinomialNB': {'alpha': [0.01, 0.1, 1, 10],
                                    'fit_prior': ['True', 'False']}}
    
    # create list to store model results
    ret = []
    
    # gridsearch function with classifiers, data and parameter grid as input
    custom_gridsearch(clfs, data, param_grid)
    
    # plot results
    plot_accuracy(ret)
    plot_f1_score(ret)
    plot_auc(ret)
    
    # print results
    print(ret)