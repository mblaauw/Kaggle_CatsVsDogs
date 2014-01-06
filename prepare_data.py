import pandas as pd
import numpy as np
import pylab as pl
from PIL import Image
import os
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier



#setup a standard image size; this will distort some images but will get everything into the same shape

# http://blog.yhathq.com/posts/image-classification-in-Python.html
# http://nbviewer.ipython.org/gist/hernamesbarbara/5768969

STANDARD_SIZE = (256, 256)

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose == True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE, Image.ANTIALIAS)
    #img = img.convert('1')
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

# test code

img_dir = "data/sample/"
images = [img_dir + f for f in os.listdir(img_dir)]
labels = ["dog" if "dog" in f.split('/')[-1] else "cat" for f in images]

# Build matrix and flatten it
# Once resized to 300x167, we flatten the images using the flatten_image function, so each is represented as a row in a 2D array.
data = []
for image in images:
    print image
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)

data = np.array(data)

# save dataset since its a long run
np.savetxt('processed_features-nparray.txt', data)



# Define a training and test set


is_train = np.random.uniform(0, 1, len(data)) <= 0.7
y = np.where(np.array(labels)=="dog", 1, 0)

train_x, train_y = data[is_train], y[is_train]
test_x, test_y = data[is_train==False], y[is_train==False]

# Creating features
# 45,000 features is a lot to deal with for many algorithms, so we need to reduce the number of dimensions somehow.
# As an example, let's transform the dataset into just 2 components which we can easily plot in 2 dimensions.
pca = RandomizedPCA(n_components=2)
X = pca.fit_transform(data)
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, "dog", "cat")})
colors = ["red", "yellow"]
for label, color in zip(df['label'].unique(), colors):
    mask = df['label']==label
    pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
pl.legend()
#pl.show()