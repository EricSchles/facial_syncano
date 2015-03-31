import cv2

image = cv2.imread("opencv_logo.png")
resized_image = cv2.resize(image, (100, 50)) 
cv2.imwrite("smaller_logo.png",resized_image)
