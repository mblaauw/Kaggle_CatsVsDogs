library(biOps)

folder = 'data/samples/'
files  = dir(folder, pattern = "^[a-lr]", full.names = TRUE, ignore.case = TRUE)



buildClassificationMatrix <- function ( filename=file, classtype=class.type ) {
  # main image
  image.rgb = readJpeg(filename)
  # convert to gray
  #image.gray= imgRGB2Grey(image.rgb)
  
  # generate KMeans image
  image.KMeans  = imgKDKMeans(imgdata=image.rgb,k=12,maxit=20)
  # generate EKMeans image - unsupervised classiﬁcation through the k-means
  image.EKMeans  = imgKDKMeans(imgdata=image.rgb,k=12,maxit=20)
  # generate histogram
  image.hist    = imgHistogram(image.EKMeans,col='red')
  
  x = as.data.frame(t(image.hist$density),stringsAsFactors=FALSE)
  result = data.frame(c(type=classtype,x))
  
  result 
}


for (i in 1:length(files)) {
  file = files[i]
  # establish cat or dog from filename
  class.type = substr(file,start=14,stop=16)
  
  result = buildClassificationMatrix(filename=file, classtype=class.type)  

  
  if (i == 1 ) {
    #finalresult = result
    print(ncol(result))
  } else {
    #finalresult = rbind(finalresult, result)    
    print(ncol(result))
  }
 
    
  
}



