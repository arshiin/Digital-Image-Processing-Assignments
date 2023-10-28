import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# PART A
img = cv.imread('Input_data/Head-MRI.tif', 0)
plt.figure()

plt.subplot(2, 1, 1)
img_uint8 = np.uint8(img)
plt.title('uint8')
plt.imshow(img_uint8, cmap='gray')

plt.subplot(2, 1, 2)
img_array = np.array(img)  # Convert the image to an array
img_float = np.float64(img_array) / 255.0
plt.title('float')
plt.imshow(img_float, cmap='gray', vmin=0, vmax=1)
plt.tight_layout()  # We put this to prevent the titles overlapping

# PART B
row150 = img[149:150]
row180 = img[179:180]
plt.figure()

plt.subplot(2, 1, 1)
plt.title('Histograms')
# Since we have to plot the histogram with the "plt.plot" command, we plot it using OpenCV histogram command
hist150 = cv.calcHist([row150], [0], None, [256], [0, 256])
hist180 = cv.calcHist([row180], [0], None, [256], [0, 256])
plt.plot(hist150, color='r', label='Row 150')
plt.plot(hist180, color='b', label='Row 180')
plt.legend()
plt.tight_layout()

# PART C
plt.subplot(2, 1, 2)
plt.title('Rows 150 and 180')
# We color the whole image except the required rows white so the only visible parts of the image would be the two rows
img[0:149] = 255
img[150:179] = 255
img[180:] = 255
plt.imshow(img, cmap='gray')
plt.axis(False)
plt.tight_layout()
plt.show()
