import cv2
import numpy as np
import math


# Load the image
image = cv2.imread('images/an.jpeg')



# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to obtain binary image
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store similar object pairs
similar_pairs = []

# Iterate over contours to compare object similarity
for i in range(len(contours)):
    for j in range(i+1, len(contours)):
        # Extract the bounding rectangles for the current pair of objects
        rect1 = cv2.boundingRect(contours[i])
        rect2 = cv2.boundingRect(contours[j])

        # Calculate the similarity score based on the bounding rectangle area ratio
        similarity = cv2.matchShapes(contours[i], contours[j], cv2.CONTOURS_MATCH_I2, 0)

        # Define a threshold for determining similar objects
        similarity_threshold = 0.1

        # If similarity score is below the threshold, consider the objects similar
        if similarity < similarity_threshold:
            similar_pairs.append((rect1, rect2))

# Draw bounding boxes and label similar object pairs
distance=[]

for pair in similar_pairs:
    rect1, rect2 = pair
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Draw the bounding rectangles for similar objects
    cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
    
#     if y2-y1==0:
#         distance.append(0.0001)
#     else:
#         dist=math.sqrt(((x2-x1)*(x2-x1))/((y2-y1)*(y2-y1)))
#         distance.append(dist)
    

# distance=[i/max(distance) for i in distance]
# i=0
# for j in distance:
#     if j>=0.7:
#         print(i)
#         rect1, rect2 = similar_pairs[i]
#         x1, y1, w1, h1 = rect1
#         x2, y2, w2, h2 = rect2
#         cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
#         cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
#     i+=1
         


print(len(similar_pairs))
print(len(distance))

print("Similar Pairs ------ ",max(similar_pairs))
# Display the labeled image
cv2.imshow("Labeled Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()