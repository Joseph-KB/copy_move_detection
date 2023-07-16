import cv2
import numpy as np

def calculate_phash(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 32x32 pixels
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    
    # Calculate the Discrete Cosine Transform (DCT)
    dct = cv2.dct(np.float32(resized))
    
    # Keep only the top-left 8x8 coefficients
    dct_low_freq = dct[:8, :8]
    
    # Calculate the median value of the coefficients (excluding the DC coefficient)
    median = np.median(dct_low_freq[1:, 1:])
    
    # Set the bits of the hash based on the coefficients' values
    phash = 0
    for i in range(dct_low_freq.shape[0]):
        for j in range(dct_low_freq.shape[1]):
            if dct_low_freq[i, j] > median:
                phash |= 1 << (i * 8 + j)
    
    return phash

def hamming_distance(hash1, hash2):
    # Calculate the Hamming distance between two hashes
    return bin(hash1 ^ hash2).count('1')

def detect_copy_move(image, block_size=16, threshold=10):
    # Load the image
    
    # Calculate the image's phash
    phash = calculate_phash(image)
    
    # Split the image into blocks
    blocks = []
    for i in range(0, image.shape[0] - block_size + 1, block_size):
        for j in range(0, image.shape[1] - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append((block, (i, j)))
    
    # Compare each block's phash with other blocks' phashes
    matches = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            hash1 = calculate_phash(blocks[i][0])
            hash2 = calculate_phash(blocks[j][0])
            distance = hamming_distance(hash1, hash2)
            
            if distance <= threshold:
                matches.append((blocks[i][1], blocks[j][1]))
    
    return matches

# Example usage
image_path = 'images/anco.jpeg'
image=cv2.imread(image_path)

matches = detect_copy_move(image)

if len(matches) > 0:
    print("Copy-move forgery detected!")
    for match in matches:
        print("Block at ({}, {}) is similar to block at ({}, {})".format(
            match[0][0], match[0][1], match[1][0], match[1][1]))
else:
    print("No copy-move forgery detected.")









# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread("images/an.jpeg")

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to obtain binary image
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# # Find contours in the binary image
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Initialize a list to store similar object pairs
# similar_pairs = []

# # Iterate over contours to compare object similarity
# for i in range(len(contours)):
#     for j in range(i + 1, len(contours)):
#         # Extract the bounding rectangles for the current pair of objects
#         rect1 = cv2.boundingRect(contours[i])
#         rect2 = cv2.boundingRect(contours[j])

#         # Calculate the similarity score based on the bounding rectangle area ratio
#         similarity = cv2.matchShapes(contours[i], contours[j], cv2.CONTOURS_MATCH_I2, 0)

#         # Define a threshold for determining similar objects
#         similarity_threshold = 0.1

#         # If similarity score is below the threshold, consider the objects similar
#         if similarity < similarity_threshold:
#             similar_pairs.append((rect1, rect2))

# # Draw bounding boxes and label similar object pairs
# for pair in similar_pairs:
#     rect1, rect2 = pair
#     x1, y1, w1, h1 = rect1
#     x2, y2, w2, h2 = rect2

#     # Draw the bounding rectangles for similar objects
#     cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
#     cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)


# # Display the labeled image
# cv2.imshow("Copy-Move Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()