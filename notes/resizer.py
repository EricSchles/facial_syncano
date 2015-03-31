import cv2
from sys import argv

img = argv[1]
image = cv2.imread(img)
resized_image = cv2.resize(image, (100, 50)) 
cv2.imwrite(img,resized_image)
