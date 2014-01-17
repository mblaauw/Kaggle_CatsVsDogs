__author__ = 'MICH'
import os
import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import expon
from sklearn.decomposition import RandomizedPCA
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from time import time
from sklearn.grid_search import RandomizedSearchCV

wd = 'C:/03_P-PROJECTS/Kaggle_CatsVsDogs/' #change this to make the code work
dataTrainDir = 'C:/03_P-PROJECTS/Kaggle_CatsVsDogs/Data/Data/train/'
dataTestDir = 'C:/03_P-PROJECTS/Kaggle_CatsVsDogs/Data/test/'

os.chdir(wd)

labels = ['cat.', 'dog.']
desiredDimensions = [30, 30]

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
bigMatrix = lil_matrix((len(indexesIm) + len(testIndexes), desiredDimensions[0] * desiredDimensions[1]))

for i in range(len(indexesIm)):
    bigMatrix[i, :] = preprocessImg(animalInput(y[i]), idxImages[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

someNumbers = range(mTest)
for ii in someNumbers:
    bigMatrix[testIndexes[ii], :] = preprocessImg(animalInput('printNothing'), ii + 1, desiredDimensions[0], desiredDimensions[1], dataTestDir)

#Transform to csr matrix
bigMatrix = bigMatrix.tocsr()

#Reduce features to main components so that they contain 99% of variance
pca = RandomizedPCA(n_components=150, whiten = True)
pca.fit(bigMatrix)
varianceExplained = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)

def anonFunOne(vector):
    variance = 0
    for ii in range(len(vector)):
            variance += vector[ii]
            if variance > 0.99:
                componentIdx = ii
                return(componentIdx)
            break

pca = RandomizedPCA(n_components=150, whiten = True)
BigMatrixReduced = pca.fit_transform(bigMatrix, y = anonFunOne(varianceExplained))

#Divide train Matrix and Test Matrix (for which I don't have labels)
trainMatrixReduced = BigMatrixReduced[0:max(indexesIm), :]
testMatrixReduced = BigMatrixReduced[testIndexes[0]:BigMatrixReduced.shape[0], :]

#Divide dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    trainMatrixReduced, y[0:24999], test_size=0.4, random_state=0) #fix this

#random grid search of hiperparameters

#create a classifier
clf = svm.SVC(verbose = True)

# specify parameters and distributions to sample from
params2Test = {'C': expon(scale=100), 'gamma': expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['auto']}

#run randomized search
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions = params2Test, n_iter = n_iter_search)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
type(random_search.grid_scores_)

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
clf = svm.SVC(verbose = True)
clf.fit(trainMatrixReduced, y[0:24999]) #fix this

#Prediction
#predictionFromTest = clf.predict_proba(testMatrixReduced)
predictionFromTest = clf.predict(testMatrixReduced)
#label = predictionFromTest[:, 1]
idVector = range(1, mTest + 1)

#predictionsToCsv = np.column_stack((idVector, label))
predictionsToCsv = np.column_stack((idVector, predictionFromTest))

import csv

ofile = open('predictionIII.csv', "wb")
fileToBeWritten = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

for row in predictionsToCsv:
    fileToBeWritten.writerow(row)

ofile.close()