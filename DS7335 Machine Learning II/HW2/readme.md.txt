This homework was to create a custom gridsearch function.

# This function takes a clf_dictionary (classifier), data, and a clf_hyperparameter
# dictionary as inputs. It will create a train/test split using the input
# data and then for each classifier, it will cycle through every combination 
# of hyperparameters and return the best params based on f1_score. 
# The accuracy, f1_score and auc will be outputted next to the name of each
# classifier. The metrics for each model is saved in a list and used
# for plotting.
# This function will test logistic regression, random forest, KNN, SVM, 
# decision tree classifier, xgb, gaussian NB and multinomial NB models.