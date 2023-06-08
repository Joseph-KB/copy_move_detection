import cv2
import numpy as np

def detect_copy_move(image_path, cluster_threshold=50, similarity_threshold=0.7):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain binary image
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract feature descriptors for each contour using ORB
    orb = cv2.ORB_create()
    keypoints = []
    descriptors = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        
        # Resize the ROI to a non-zero size
        roi = cv2.resize(roi, (256, 256))

        kp, desc = orb.detectAndCompute(roi, None)
        keypoints.append(kp)
        descriptors.append(desc)

    # Match feature descriptors to identify similar objects
    matches = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    for i in range(len(descriptors)):
        for j in range(i+1, len(descriptors)):
            matches_knn = bf.knnMatch(descriptors[i], descriptors[j], k=2)
            good_matches = []
            for m, n in matches_knn:
                if m.distance < similarity_threshold * n.distance:
                    good_matches.append(m)
            if len(good_matches) > cluster_threshold:
                matches.append((i, j))

    # Create a mask to mark the copied regions
    mask = np.zeros_like(gray)

    # Draw bounding rectangles around the copied regions and mark them in the mask
    for match in matches:
        contour1 = contours[match[0]]
        contour2 = contours[match[1]]
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        mask[y1:y1+h1, x1:x1+w1] = 255
        mask[y2:y2+h2, x2:y2+w2] = 255

    # Display the original image and the copied regions mask
    cv2.imshow('Original Image', image)
    cv2.imshow('Copied Regions Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to the image
image_path = 'images/an.jpeg'

# Detect copy-move forgery for different objects in the image
detect_copy_move(image_path)







#    'images/prjct.jpg'