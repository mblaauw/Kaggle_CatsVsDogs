__author__ = 'MICH'


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



# RandomizedPCA in 5 dimensions
#
# Instead of 2 dimenisons, we're going to do RandomizedPCA in 5 dimensions.
# This will make it a bit harder to visualize, but it will make it easier for
# some of the classifiers to work with the dataset.
pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


# This gives our classifier a nice set of tabular data that we can then use to train the model
train_x[:5]


# We're going to be using a K-Nearest Neighbors classifier. Based on our set of
# training data, we're going to caclulate which training obersvations are closest
# to a given test point. Whichever class has the most votes wins.
knn = KNeighborsClassifier()
knn.fit(train_x, train_y)

pd.crosstab(test_y, knn.predict(test_x), rownames=["Actual"], colnames=["Predicted"])

from yhat import Yhat, BaseModel

class ImageClassifier(BaseModel):
    def require(self):
        from StringIO import StringIO
        from PIL import Image
        import base64

    def transform(self, image_string):
    	#we need to decode the image from base64
    	image_string = base64.decodestring(image_string)
        #since we're seing this as a JSON string, we use StringIO so it acts like a file
    	img = StringIO(image_string)
        img = Image.open(img)
        img = img.resize(self.STANDARD_SIZE)
        img = list(img.getdata())
        img = map(list, img)
        img = np.array(img)
        s = img.shape[0] * img.shape[1]
        img_wide = img.reshape(1, s)
        return self.pca.transform(img_wide[0])

    def predict(self, data):
        preds = self.knn.predict(data)
        preds = np.where(preds==1, "check", "drivers_license")
        pred = preds[0]
        return {"image_label": pred}


img_clf = ImageClassifier(pca=pca, knn=knn, STANDARD_SIZE=STANDARD_SIZE)

# authenticate
yh = Yhat("YOUR USERNAME", "YOUR API KEY")

# upload model to yhat
yh.upload("imageClassifier", img_clf)


#call the api w/ some example data
# this one is a drivers license
new_image = open("dl16.jpeg", 'rb').read()

import base64
#we need to make the image JSON serializeable
new_image = base64.encodestring(new_image)

print yh.predict("imageClassifier", version=4, data=new_image)