__author__ = 'MBlaauw'


#import libraries
import os
import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import expon
from sklearn.decomposition import RandomizedPCA
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from time import time
from sklearn.grid_search  import GridSearchCV
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

wd = '/Users/mblaauw/Downloads/06_P_PROJECTS/Kaggle_CatsVsDogs/' #change this to make the code work

dataTrainDir = '/Users/mblaauw/Downloads/06_P_PROJECTS/Kaggle_CatsVsDogs/data/train/'
dataTestDir = '/Users/mblaauw/Downloads/06_P_PROJECTS/Kaggle_CatsVsDogs/data/test1/'


os.chdir(wd)

labels = ['cat.', 'dog.']
desiredDimensions = [45, 45]


#define loading and pre-processing function grayscale
def preprocessImg(animal, number, dim1, dim2, dataDir):
    imageName = '{0:s}{1:s}{2:d}{3:s}'.format(dataDir, animal, number, '.jpg')
    npImage = cv2.imread(imageName)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
    avg = np.mean(npImage.reshape(1, npImage.shape[0] * npImage.shape [1]))
    avg = np.tile(avg, (npImage.shape[0], npImage.shape [1]))
    npImage = npImage - avg
    npImage = cv2.resize(npImage, (dim1, dim2))
    return(npImage.reshape(1, dim1 * dim2))


#m = 1000 #pet Train dataset
m = 12500 #full Train dataset
mTest = 12500 #number of images in the test set

indexesIm = np.random.permutation(m * len(labels))
idxImages = np.tile(range(m), len(labels))
idxImages = idxImages[indexesIm]
testIndexes = range(len(indexesIm), len(indexesIm) + mTest)
y = np.append(np.tile(0, m), np.tile(1, m))
y = y[indexesIm]


def animalInput(theNumber):
    if theNumber == 0:
        return 'cat.'
    elif theNumber == 1:
        return 'dog.'
    else:
        return ''

#Build the sparse matrix with the preprocessed image data for both train and test data
bigMatrixTrain = np.empty(shape=(((len(indexesIm), desiredDimensions[0] * desiredDimensions[1]))))
bigMatrixTest = np.empty(shape=(((len(testIndexes), desiredDimensions[0] * desiredDimensions[1]))))

for i in range(len(indexesIm)):
    bigMatrixTrain[i, :] = preprocessImg(animalInput(y[i]), idxImages[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

someNumbers = range(mTest)
for ii in someNumbers:
    bigMatrixTest[ii, :] = preprocessImg(animalInput('printNothing'), ii + 1, desiredDimensions[0], desiredDimensions[1], dataTestDir)

#Divide dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    bigMatrixTrain, y, test_size = 0.4, random_state = 0) #fix this

#Reduce features to main components so that they contain 99% of variance
n_components = 250

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components = n_components, whiten = True)
pca.fit(X_train)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 1e4, 1e5],
              'gamma': [0.0001,0.001, 0.01, 0.1], }
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='auto', verbose = True, probability=False), param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
t0 = time()
prediction = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

correctValues = sum(prediction == y_test)
percentage = float(correctValues)/len(y_test)

print(percentage)

#mmodel number 2
bigMatrixTrain = (bigMatrixTrain - np.min(bigMatrixTrain, 0)) / (np.max(bigMatrixTrain, 0) + 0.0001)  # 0-1 scaling
#Divide dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    bigMatrixTrain, y, test_size = 0.4, random_state = 0) #fix this

# specify parameters and distributions to sample from
# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger fitting time
rbm.n_components = 300
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, y_train)

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(y_test, classifier.predict(X_test))))
print("Logistic regression using RBM features:\n%s\n" % (
    confusion_matrix(y_test, classifier.predict(X_test))))


#mmodel number 3
#Divide dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    bigMatrixTrain, y, test_size = 0.4, random_state = 0) #fix this

# specify parameters and distributions to sample from
# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger fitting time
rbm.n_components = 300
logistic.C = 6000.0

n_components = 250

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components = n_components, whiten = True)
pca.fit(X_train)
print("done in %0.3fs" % (time() - t0))

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# need fix
bigMatrixTrain = (np.row_stack(X_train, X_test) - np.min(np.row_stack(X_train, X_test), 0)) / (np.max(np.row_stack(X_train, X_test), 0) + 0.0001)  # 0-1 scaling

X_train = bigMatrixTrain[0:X_train.shape[0], :]
X_test = bigMatrixTrain[X_train.shape[0] + 1 :bigMatrixTrain.shape[0], :]

# Training RBM-Logistic Pipeline
classifier.fit(X_train, y_train)

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(y_test, classifier.predict(X_test))))


print("Logistic regression using RBM features:\n%s\n" % (
    confusion_matrix(y_test, classifier.predict(X_test))))


#prediction probability
predictionFromDataset2 = clf.predict_proba(X_test)
predictionFromDataset2 = predictionFromDataset2[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictionFromDataset2)
predictionProbability = metrics.auc(fpr, tpr)

#Prediction
#predictionFromTest = clf.predict_proba(testMatrixReduced)
predictionFromTest = clf.predict(bigMatrixTest)
#label = predictionFromTest[:, 1]
idVector = range(1, mTest + 1)

#predictionsToCsv = np.column_stack((idVector, label))
predictionsToCsv = np.column_stack((idVector, predictionFromTest))

import csv

ofile = open('predictionVII.csv', "wb")
fileToBeWritten = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

for row in predictionsToCsv:
    fileToBeWritten.writerow(row)

ofile.close()