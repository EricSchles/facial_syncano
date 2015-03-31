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
    
    
