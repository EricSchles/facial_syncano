import cv2
from glob import glob
imgs = glob("*.jpg")

for img in imgs:
    image = cv2.imread(img)
    resized_image = cv2.resize(image, (100, 50)) 
    cv2.imwrite(img,resized_image)
