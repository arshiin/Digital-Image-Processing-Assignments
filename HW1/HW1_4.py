import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# PART A
img_cube = cv.imread('Input_data/Cube.tif')
cv.imshow('Original image', img_cube)
print('Image dimensions:', img_cube.shape)

# PART B
img_gray = cv.cvtColor(img_cube, cv.COLOR_BGR2GRAY)
print('Gray image dimensions:', img_gray.shape, '\nImage datatype:', img_cube.dtype)


# PART C
def func(image, bits):
    image = ((image / 255) * (2 ** bits - 1)).astype(np.uint8)
    image = (image * (255 / (2 ** bits - 1))).astype(np.uint8)

    return image


# PART D
plt.figure()
plt.suptitle('Cube image in gray with different intensities')
plt.subplot(2, 3, 1)
plt.title('8 bits')
plt.imshow(func(img_gray, 8), cmap='gray')
plt.axis(False)

plt.subplot(2, 3, 2)
plt.title('5 bits')
plt.imshow(func(img_gray, 5), cmap='gray')
plt.axis(False)

plt.subplot(2, 3, 3)
plt.title('3 bits')
plt.imshow(func(img_gray, 3), cmap='gray')
plt.axis(False)

plt.subplot(2, 3, 4)
plt.title('2 bits')
plt.imshow(func(img_gray, 2), cmap='gray')
plt.axis(False)

plt.subplot(2, 3, 5)
plt.title('1 bits')
plt.imshow(func(img_gray, 1), cmap='gray')
plt.axis(False)

# PART E
ret, img_thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
plt.subplot(2, 3, 6)
plt.title('Binary threshold')
plt.imshow(img_thresh, cmap='gray')
plt.axis(False)

# PART F
plt.savefig('9733037-3.png')

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
