import cv2

# Load the pedestrian detection HOG model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the image
image = cv2.imread('images/an.jpeg')

# Resize the image for faster processing
image = cv2.resize(image, (640, 480))

# Detect pedestrians in the image
boxes, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

# Draw bounding boxes around the detected pedestrians
for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image
cv2.imshow('Pedestrian Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
