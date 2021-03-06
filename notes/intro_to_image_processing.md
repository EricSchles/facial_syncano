#Introduction to image processing

Image processing is about applying mathematics to pictures.  It's no different than any other part of computer science, except the data structures are different than many other domains.

Image processing, or at least the techniques we'll see today cover much of the same material as any machine learning course:

* Cleaning the data:
  * filtering
* Finding patterns in the data:
 * Facial recognition
 * object recognition
 * scene analysis

##Getting data into OpenCV

Before we start with any of the material first we'll need to understand how to read in images and write out images with OpenCV:

###Reading/Writing images:

read_write.py
```
import cv2
img = cv2.imread("opencv_logo.png")
cv2.imwrite("opencv_logo_copy.png",img)
```

##Cleaning our data - Filtering

Before we can begin to apply techniques such as facial recognition, we need to make sure our data is good enough to be used by our models.

For this we'll need to be able to filter our data.

###Definition

Filtering is the process of taking an image as input and performing a set of mathematical operations to change the image in a way that is more useable programmatically.

####Example 1 - gray scaling

grayscaling.py:

```
import cv2
image = cv2.imread('opencv_logo.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png',gray_image)
```

[gray scale reference](http://stackoverflow.com/questions/12752168/why-we-should-use-gray-scale-for-image-processing)

Gray scaling is important and useful for a number of reasons.  The easiest to understand is - it makes processing images much, much faster.

####Example 2 - resizing


```
import cv2

image = cv2.imread("opencv_logo.png")
resized_image = cv2.resize(image, (100, 50)) 
cv2.imwrite("smaller_logo.png",resized_image)
```

Resizing images is useful again, for a number of reasons.  The easiest to understand is - it makes processing images much, much faster.

####Example 3 - Smoothing

smoothing.py:

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
```

[Reference for the code](http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)

As you can see making use of OpenCV in python is extremely easy.  Here we doa 2D filter on the image.  Here we are doing a simple 2D convolution.  We do this by applying a kernel, which is just a matrix and then we apply it to each pixel with a 5x5 window around the pixel that is averaged.  The overall effect is the image is blurred, as show in the results of running the filter over the original image.

Other methods of filtering include:

* `blur = cv2.blur(img,(5,5))`
* `blur = cv2.GaussianBlur(img,(5,5),0)`
* `median = cv2.medianBlur(img,5)`
* `blur = cv2.bilateralFilter(img,9,75,75)`

##Facial recognition

Now that we know how to read in our images, let's have some fun!  Facial recognition is among the greatest achievements of computation.  We glean so much information from faces - identity, emotion, age.  Our minds are made to see extremely nuanced details in faces.  Mostly because they are extremely complex, with many, many muscles that explain extremely complex emotions, without a single word.

The true power of opencv is it's ability to make the task of facial recognition easy.

face_detect.py

```
import cv2
from PIL import Image
import os
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
    
# Read the image
imagePath = os.path.abspath("person.jpg")
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(100, 100),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("faces found", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

[reference for haar cascade](http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)

Face detection with opencv2 is extremely easy.  All that is required is loading the haarcascade, and then calling detectMultiScale with a few parameters.

The haar cascade:

The notion is actually quiet simple.  The haarcascade_frontalface_default.xml file contains a ton of features that have been decided as belonging to a front facing face.  The algorithm goes through and checks each block in the picture - set with minSize for these features.  There are 6000 in total, so that's a ton of computation.  To speed things up, they also built in a detection system that looks first for areas that have no faces.  So if we detect "no face" in a given region in the picture, the detection scheme moves on or "cascades" to the next region in the image.  This allows the process to be extremely fast, extremely accurate (95% on good pictures), and extremely easy to use.

##An indepth facial recognition example:

```
import cv2
import cv
import numpy as np
from glob import glob
import os

CONFIDENCE_THRESHOLD = 100.0
ave_confidence = 0
num_recognizers = 3
recog = {}
recog["eigen"] = cv2.createEigenFaceRecognizer()
recog["fisher"] = cv2.createFisherFaceRecognizer()
recog["lbph"] = cv2.createLBPHFaceRecognizer()

#load the data initial file
filename = os.path.abspath("person.jpg")
face = cv.LoadImage(filename, cv2.IMREAD_GRAYSCALE)
face,label = face[:, :], 1

#load comparison face
compare = os.path.abspath("black_widow.jpg")
compare_face = cv.LoadImage(compare, cv2.IMREAD_GRAYSCALE)
compare_face, compare_label = compare_face[:,:], 2

images,labels = [],[]
images.append(np.asarray(face))
images.append(np.asarray(compare_face))
labels.append(label)
labels.append(compare_label)

image_array = np.asarray(images)
label_array = np.asarray(labels)
for recognizer in recog.keys():
    recog[recognizer].train(image_array,label_array)


#generate test data
test_images = glob("testing/*.jpg")
test_images = [(np.asarray(cv.LoadImage(img,cv2.IMREAD_GRAYSCALE)[:,:]),img) for img in test_images]
for t_face,name in test_images:
    t_labels = []
    for recognizer in recog.keys():
        [label, confidence] = recog[recognizer].predict(t_face)
        print "match found",name, confidence, recognizer
```

##Object recognition
    
Object detection is really just a generalized case of facial recognition.  All you need to do is is apply a different xml file and you'll get what you need.  In effort to not just show you the same technique over and over again, let's go over some other ways we can detect objects using opencv.

###Corner detection

```
import cv2
import numpy as np

filename = 'chessboard.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
```

This technique is exactly what it sounds like - making use of the an algorithm invented by Harris, we find the corners within this chessboard.

 
##References

* [brown class](http://cs.brown.edu/courses/cs143/)
