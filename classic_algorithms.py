# -*- coding: utf-8 -*-
"""
This script uses several classifiers to train 9 different models, and gives some
diagnostics about their performance. This is a first approximation for classifying
MNIST and Fashion-MNIST (must be run separately).
Holdout Method using eigth different classifiers:
    'LR ' - Logistic Regression
    'LDA' - Linear Discriminant Analysis
    'DTC' - Decision Tree Classifier
    'RFC' - Random Forest Classifier
    'ABC' - AdaBoost Classifier
    'GNB' - Gaussian NaiveBayes
    'QDA' - Quadratic Discriminant Analysis
    'SVC' - Support Vector Machine Classifier
    'MLP' - Multilayer Perceptron Classifier
To execute, put something like this in the command prompt:
python classic_algorithms.py './np_data' x_train.npy y_train.npy x_test.npy y_test.npy
"""

import numpy as np
import scipy.io as sio
import sys
import multiprocessing
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

def classification(classifier, name, x_train, x_test, y_train, y_test):
    """
        This function train a classifier with (x_train, y_train) for training and (x_test, y_test)
        for testing. It saves the final trained classifier, the score of the classifier and
        the confusion matrix.
        Args:
            classifier
                Object that contain a classifier element.
            name
                Prefix for the saved data, e.g. LDA_score.npy.
            x_train
                Matrix of 50000 samples and 32x32 features (pixels flattened)
            x_train
                Same as the previous but with 10000 samples
            y_train
                Vector which each training label corresponding to each of the 10 classes.
            y_test
                Vector which each test label corresponding to each of the 10 classes.
    """

    print ('Training Process ' + name + ' start.')

    #Fit and save in pickled file
    classifier.fit(x_train, y_train)
    fileobject = open('./diagnostics/' + name + '.p', 'w')
    pickle.dump(classifier, fileobject)
    fileobject.close()

    print ('Calculating Confusion Matrix of:  ' + name)

    temp_len = len(y_test)
    y_result = np.zeros(temp_len)

    for j in range(temp_len):
        y_result[j] = classifier.predict(x_test[j, :].reshape(1, -1))

    cm = confusion_matrix(y_test, y_result)

    np.save('./diagnostics/' + name + '_cm', cm)

    print ('Calculating Score of: ' + name)

    score = classifier.score(x_test, y_test)
    print(score)
    np.save('./diagnostics/' + name + '_score', score)

    print ('Process  ' + name + ' finish')

    return

if __name__ == '__main__':
    #main algorithm
    #we recieve 5 inputs:
    #directory: where we have our train and test files.
    #x_train_file: np matrix where every row is a list of pixels for every training image
    #y_train_file: labels for every training image
    #x_test_file: np matrix where every row is a list of pixels for every testing image
    #y_test_file: labels for every testing image

    #Inputs
    directory = sys.argv[1]
    x_train_file = sys.argv[2]
    y_train_file = sys.argv[3]
    x_test_file = sys.argv[4]
    y_test_file = sys.argv[5]

    #Open in numpy, a little bit of processing for the labels
    #Divide by 255 to standarize between 0 and 1 pixel values, standard
    #preproccessing technique
    x_train = np.load(directory + '/' + x_train_file)
    x_train = x_train/255.0
    x_test = np.load(directory + '/' + x_test_file)
    x_test = x_test/255.0
    y_train_l = np.load(directory + '/' + y_train_file)
    y_train = np.ravel(y_train_l)
    y_test_l = np.load(directory + '/' + y_test_file)
    y_test = np.ravel(y_test_l)

    print ('Loading Data set: Done')

    jobs = []

    LR = multiprocessing.Process(name='LR',
                                  target=classification,
                                  args=(LogisticRegression(),
                                        'LR',
                                        x_train, x_test, y_train, y_test))

    LDA = multiprocessing.Process(name='LDA',
                                  target=classification,
                                  args=(LinearDiscriminantAnalysis(),
                                        'LDA',
                                        x_train, x_test, y_train, y_test))

    DTC = multiprocessing.Process(name='DTC',
                                  target=classification,
                                  args=(DecisionTreeClassifier(max_depth=128),
                                        'DTC',
                                        x_train, x_test, y_train, y_test))

    RFC = multiprocessing.Process(name='RFC',
                                  target=classification,
                                  args=(RandomForestClassifier(max_depth=45,
                                                               max_features=18),
                                        'RFC',
                                         x_train, x_test, y_train, y_test))

    ABC = multiprocessing.Process(name='ABC',
                                  target=classification,
                                  args=(AdaBoostClassifier(),
                                        'ABC',
                                        x_train, x_test, y_train, y_test))

    GNB = multiprocessing.Process(name='GNB',
                                  target=classification,
                                  args=(GaussianNB(),
                                        'GNB',
                                        x_train, x_test, y_train, y_test))

    QDA = multiprocessing.Process(name='QDA',
                                  target=classification,
                                  args=(QuadraticDiscriminantAnalysis(),
                                        'QDA',
                                        x_train, x_test, y_train, y_test))

    SVM = multiprocessing.Process(name='SVM',
                                 target=classification,
                                 args=(SVC(),
                                       'SVM',
                                       x_train, x_test, y_train, y_test))

    MLP = multiprocessing.Process(name='MLP',
                                 target = classification,
                                 args=(MLPClassifier(),
                                       'MLP',
                                       x_train, x_test, y_train, y_test))

    print('Starting Parallel Processing')

    Process = [LDA, DTC, RFC, ABC, GNB, QDA, SVM, LR, MLP]
    #Process = [MLP]
    for i in Process:
        jobs.append(i)

    for i in Process:
        i.start()

    for i in Process:
        i.join()
