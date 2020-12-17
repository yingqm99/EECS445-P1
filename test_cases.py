# EECS 445 - Fall 2020
# Project 1 - test_cases.py

import pandas as pd
import numpy as np

from project1 import extract_dictionary, generate_feature_matrix

def test_dictionary():
    """
    Test case for extract_dictionary()
    """
    print('TESTING EXTRACT DICTIONARY')
    X_train = pd.DataFrame({'useful': [0],
                            'reviewText': ['BEST restaurant ever! It\'s great'],
                            'funny': [1],
                            'cool': [0],
                            'label': [1]})

    expected_dictionary = {'best': 0, 'restaurant': 1, 'ever': 2, 'it': 3, 's': 4, 'great': 5}
    dictionary = extract_dictionary(X_train)

    print('EXPECTED OUTPUT:\t' + str(expected_dictionary))
    print('STUDENT OUTPUT: \t' + str(dictionary))  
    print('SUCCESS') if dictionary == expected_dictionary else print('OUTPUTS DIFFER')

    return dictionary


def test_feature_matrix(dictionary):
    """
    Test case for generate_feature_matrix()
    """
    print('TESTING GENERATE FEATURE MATRIX')
    X_test = pd.DataFrame({'useful': [20],
                           'reviewText': ['Markley is the best dorm! Tendie Fridays are some of the best meals I have EVER had'],
                           'funny': [1],
                           'cool': [3],
                           'label': [1]})

    expected_feature_matrix = np.array([[1., 0., 1., 0., 0., 0.]])
    feature_matrix = generate_feature_matrix(X_test, dictionary)

    print('EXPECTED OUTPUT:\t' + str(expected_feature_matrix))
    print('STUDENT OUTPUT: \t' + str(feature_matrix))
    print('SUCCESS') if np.array_equal(feature_matrix, expected_feature_matrix) else print('OUTPUTS DIFFER')


def main():
    dictionary = test_dictionary()
    print('----------------')
    test_feature_matrix(dictionary)


if __name__ == '__main__':
    main()
