import matplotlib.pyplot as plt
import cv2 as cv

# PART A
img = cv.imread('Input_data/fingerprint.tif', 0)

# PART B
ret, img_bin = cv.threshold(img, 120, 255, cv.THRESH_BINARY_INV)

# PART C
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
img_opening = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
img_closing = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)

# plt.subplot(1, 3, 1)
# plt.title('Original threshold image')
# plt.imshow(img_bin, cmap='gray', vmin=0, vmax=255)
# plt.axis(False)
#
# plt.subplot(1, 3, 2)
# plt.title('Image opening')
# plt.imshow(img_opening, cmap='gray', vmin=0, vmax=255)
# plt.axis(False)
#
# plt.subplot(1, 3, 3)
# plt.title('Image closing')
# plt.imshow(img_closing, cmap='gray', vmin=0, vmax=255)
# plt.axis(False)
#
# plt.tight_layout()
# plt.show()
