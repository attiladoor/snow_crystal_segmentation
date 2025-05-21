import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/stejan/image_analysis/20180228_183056_561.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
#_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Alternatively, use edge detection (if thresholding doesn't work well)
binary = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the contour with the largest area (assuming that's the object of interest)
contour = max(contours, key=cv2.contourArea)

# Get the minimum area rectangle that encloses the contour
rect = cv2.minAreaRect(contour)

# Get the width and height of the rectangle
width, height = rect[1]

# Draw the rectangle on the image for visualization
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# Display the results
print(f"Width: {width}")
print(f"Height: {height}")

cv2.imshow("iamge with rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

