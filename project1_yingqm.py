# EECS 445 - Fall 2020
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *

def extract_dictionary(df):
    """
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    count = 0
    word_dict = {}
    for i in df['reviewText']:
        review = i
        for ch in review:
            if ch in string.punctuation:
                review = review.replace(ch, " ")
        words = review.split()
        for word in words:
            if word.lower() not in word_dict:
                word_dict[word.lower()] = count
                count = count + 1
    # TODO: Implement this function
    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (# of reviews, # of words in dictionary).
    Input:
        df: dataframe that has the ratings and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    row = 0
    for i in df['reviewText']:
        review = i
        for ch in review:
            if ch in string.punctuation:
                review = review.replace(ch, " ")
        words = review.split()
        for word in words:
            if word.lower() in word_dict:
                feature_matrix[row][word_dict[word.lower()]] = 1
        row = row + 1
    # TODO: Implement this function
    return feature_matrix

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    if metric == "AUROC":
        return np.float64(metrics.roc_auc_score(y_true, y_pred))
    if metric == "f1-score":
        return np.float64(metrics.f1_score(y_true, y_pred, zero_division=1))
    if metric == "accuracy":
        return np.float64(metrics.accuracy_score(y_true, y_pred))
    if metric == "precision":
        # return np.float64(metrics.precision_score(y_true, y_pred))
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        return np.float64(confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]))
    if metric == "sensitivity":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        return np.float64(confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]))
    if metric == "specificity":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        return np.float64(confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1]))
    
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    
    # TODO: Implement this function
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful
    
    skf = StratifiedKFold(n_splits = k, shuffle = False)
            

    #Put the performance of the model on each fold in the scores array
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = []
        if metric == "AUROC":
            y_pred = clf.decision_function(X_test)
        if metric != "AUROC":
            y_pred = clf.predict(X_test)
        scores.append(performance(y_test, y_pred, metric = metric))
        

    #And return the average performance across all fold splits.
    return np.array(scores).mean()

def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.
    return LinearSVC(penalty = penalty, C=c, class_weight = class_weight)

def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    best_C_val=0.0
    # TODO: Implement this function
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM
    for c in C_range:
        clf = select_classifier(penalty = penalty, c = c, degree = 1)
        current_performance = cv_performance(clf=clf, X=X, y=y, k=k, metric=metric)
        # print(current_performance)
        if current_performance > best_C_val:
            best_C_val = current_performance
    return best_C_val


def plot_weight(X,y,penalty,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []
        
    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)

    # for c in C_range:


    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            param_range: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter values for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance as a pair (C,r)
    """
    best_C_val,best_r_val = 0.0, 0.0
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    return best_C_val,best_r_val

def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    
    

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)
    # TODO: Questions 2, 3, 4
    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    df = load_data('data/dataset.csv')
    word_dict = extract_dictionary(df)
    print(len(word_dict))
    print(np.sum(X_train) / X_train.shape[0])
    # 3.1.d
    print(select_param_linear(X=X_train, y=Y_train, k=5, metric="accuracy", C_range=C_range))
    print(select_param_linear(X=X_train, y=Y_train, k=5, metric="f1-score", C_range=C_range))
    print(select_param_linear(X_train, Y_train, 5, "AUROC", C_range))
    print(select_param_linear(X_train, Y_train, 5, "precision", C_range))
    print(select_param_linear(X_train, Y_train, 5, "sensitivity", C_range))
    print(select_param_linear(X_train, Y_train, 5, "specificity", C_range))

    # 3.1.e

    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)


if __name__ == '__main__':
    main()
