library(biOps)
library(biOpsGUI)

file = 'data/samples/dog.1.jpg'

image = readJpeg(file)


# establish cat or dog from filename
class.type = substr(file,start=14,stop=16)

# generate KMeans image
image.KMeans  = imgKDKMeans(imgdata=image,k=10,maxit=20)
# generate EKMeans image - unsupervised classiﬁcation through the k-means
image.EKMeans  = imgKDKMeans(imgdata=image,k=10,maxit=20)
# generate histogram
image.hist    = imgHistogram(image.KMeans,col='red')




plot.imagedata(image.EKMeans)
