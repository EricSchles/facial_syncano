import os
import cv
import cv2
import numpy as np
from glob import glob
CASCADE = "./haarcascade_frontalface_alt.xml"
face_dir = "./face_root_directory/"

global min_size = (20, 20)
global IMAGE_SCALE = 2
global haar_scale = 1.2
global min_neighbors = 3
global haar_flags = 0
global label_dict = {}
global variable_faces = []
import random
import pickle
from data import get_picture
import requests
from PIL import Image
from StringIO import StringIO

def load_face(recognizers):
    url = get_picture(pickle.load(open("credentials.p","rb")))[0]["image"]["image"]
    r = requests.get(url,stream=True)
    i = Image.open(StringIO(r.content))
    i.save("image.png")
    full_path = os.path.abspath("image.png")
    id_counter = 0
    print "got here"
    for recognizer in recognizers:
        label_dict[recognizer][id_counter] = person
    print "got here"
    face = cv.LoadImage(full_path,cv2.IMREAD_GRAYSCALE)
    return face[:, :],id_counter
    
def train_recognizers(recognizers):
    for recognizer in recognizers:
        label_dict[recognizer] = {}
    images = []
    labels = []

    face,id_counter = load_face(recognizers)
    
    images.append(np.asarray(face))
    labels.append(id_counter)
    for recognizer in recognizers:
        image_copy = list(images)
        label_copy = list(labels)
        image_copy.append(np.asarray(face))
        label_copy.append(id_counter)

        image_array = np.asarray(image_copy)
        label_array = np.asarray(label_copy)
        recognizer.train(image_array, label_array)
    return recognizers


if __name__ == '__main__':
    num_iterations = 300
    for j in xrange(num_iterations):
        try:
            num_recognizers = 3
            recognizers = []
            for j in xrange(num_recognizers):
                lbh_recognizer = cv2.createLBPHFaceRecognizer()
                recognizers.append(lbh_recognizer)
                recognizers = train_recognizers(recognizers)
            print "got here"
            first_item = True
            matches_this_iteration = 0
            false_positives_this_iteration = 0
            average_confidence = 0
            CONFIDENCE_THRESHOLD = 100.0
            labels = []
            for lbh_recognizer in recognizers:
                [label, confidence] = lbh_recognizer.predict(np.asarray(face))
                average_confidence += confidence
                labels.append(label)
                average_confidence /= num_recognizers
        except:
            pass
    print "is how likely this is a match"
    
