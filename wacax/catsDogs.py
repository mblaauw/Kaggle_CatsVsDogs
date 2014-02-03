__author__ = 'MBlaauw'
#version 0.3
#the revenge of the ng

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


wd = '/Users/mblaauw/Downloads/06_P_PROJECTS/Kaggle_CatsVsDogs/' #change this to make the code work

dataTrainDir = '/Users/mblaauw/Downloads/06_P_PROJECTS/Kaggle_CatsVsDogs/data/train/'
dataTestDir = '/Users/mblaauw/Downloads/06_P_PROJECTS/Kaggle_CatsVsDogs/data/test1/'


os.chdir(wd)

labels = ['cat.', 'dog.']
desiredDimensions = [30, 30]

#define loading and pre-processing function grayscale
def preprocessImg(animal, number, dim1, dim2, dataDir):
    imageName = '{0:s}{1:s}{2:d}{3:s}'.format(dataDir, animal, number, '.jpg')
    npImage = cv2.imread(imageName)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
    #vectorof255s =  np.tile(255., (npImage.shape[0], npImage.shape [1]))
    #npImage = np.divide(npImage, vectorof255s)
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
bigMatrix = lil_matrix((len(indexesIm) + len(testIndexes), desiredDimensions[0] * desiredDimensions[1]))

for i in range(len(indexesIm)):
    bigMatrix[i, :] = preprocessImg(animalInput(y[i]), idxImages[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

someNumbers = range(mTest)
for ii in someNumbers:
    bigMatrix[testIndexes[ii], :] = preprocessImg(animalInput('printNothing'), ii + 1, desiredDimensions[0], desiredDimensions[1], dataTestDir)

#Transform to csr matrix and standarization
bigMatrix = bigMatrix.tocsr()
bigMatrix = preprocessing.scale(bigMatrix, with_mean=False)

#extract features with neural nets (Restricted Boltzmann Machine)
#RBM = BernoulliRBM(verbose = True)
#RBM.learning_rate = 0.06
#RBM.n_iter = 20
#RBM.n_components = 100
#min_max_scaler = preprocessing.MinMaxScaler()
#RBM.fit(min_max_scaler.fit_transform((bigMatrix.todense()))

#Reduce features to main components so that they contain 99% of variance
pca = RandomizedPCA(n_components=250, whiten = True)
pca.fit(bigMatrix)
varianceExplained = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)

def anonFunOne(vector):
    variance = 0
    for ii in range(len(vector)):
            variance += vector[ii]
            if variance > 0.99:
                return (ii)
                break

bigMatrix = pca.fit_transform(bigMatrix, y = anonFunOne(varianceExplained))

#Divide train Matrix and Test Matrix (for which I don't have labels)
trainMatrixReduced = bigMatrix[0:max(indexesIm), :]
testMatrixReduced = bigMatrix[testIndexes[0]:bigMatrix.shape[0], :]

#Divide dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    trainMatrixReduced, y[0:24999], test_size=0.4, random_state=0) #fix this

#random grid search of hiperparameters
#create a classifier
clf = svm.SVC(verbose = True)

# specify parameters and distributions to sample from
params2Test = {'C': [1, 3, 10, 30, 100, 300], 'gamma': [0.001], 'kernel': ['rbf']}

#run randomized search
grid_search = GridSearchCV(clf, param_grid = params2Test)

start = time()
grid_search.fit(trainMatrixReduced, y[0:24999])
print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(grid_search.grid_scores_)))
type(grid_search)
grid_search.grid_scores_

#Machine Learning part
#Support vector machine model
clf.fit(X_train, y_train)

#prediction
predictionFromDataset = clf.predict(X_test)

correctValues = sum(predictionFromDataset == y_test)
percentage = float(correctValues)/len(y_test)

print(percentage)

#prediction probability


predictionFromDataset2 = clf.predict_proba(X_test)
predictionFromDataset2 = predictionFromDataset2[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictionFromDataset2)
predictionProbability = metrics.auc(fpr, tpr)

#Predict images from the test set
#Train the model with full data set
clf = svm.SVC(C = 10, gamma = 0.001, kernel= 'rbf',verbose = True)
clf.fit(trainMatrixReduced, y[0:24999]) #fix this

#Prediction
#predictionFromTest = clf.predict_proba(testMatrixReduced)
predictionFromTest = clf.predict(testMatrixReduced)
#label = predictionFromTest[:, 1]
idVector = range(1, mTest + 1)

#predictionsToCsv = np.column_stack((idVector, label))
predictionsToCsv = np.column_stack((idVector, predictionFromTest))

import csv

ofile = open('predictionVI.csv', "wb")
fileToBeWritten = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

for row in predictionsToCsv:
    fileToBeWritten.writerow(row)

ofile.close()