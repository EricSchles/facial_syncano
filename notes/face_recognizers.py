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
