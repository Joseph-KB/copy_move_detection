import cv2
import numpy as np

def detect_copy_move_forgery(image_path, block_size=1, threshold=0.9):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the block-wise mean values
    mean_blocks = cv2.boxFilter(gray, cv2.CV_32F, (block_size, block_size))

    # Threshold the mean blocks to identify potential duplicated regions
    _, binary = cv2.threshold(mean_blocks, threshold * 255, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to enhance the detected regions
    binary = cv2.dilate(binary, None)
    binary = cv2.erode(binary, None)

    # Find contours of the potential duplicated regions
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the detected duplicated regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the output image
    cv2.imshow('Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'images/an.jpeg'
detect_copy_move_forgery(image_path)
