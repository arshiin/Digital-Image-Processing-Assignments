import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# PART B
def transform2(img, A, B):
    h, w = img.shape
    img2 = np.zeros((h, w))  # create an empty matrix to generate the new transformed image in it
    for i in range(h):
        for j in range(w):
            if not A < img[i][j] < B:
                img[i][j] == 0
                img2[i][j] = img2[i][j] + img[i][j]
    return img2


# PART C
HeadCT = cv.imread('Input_data/HeadCT.tif', 0)
t_HeadCT = transform2(HeadCT, 10, 60)  # the surmised intensities that do the desired transformation

plt.figure()
plt.subplot(1, 2, 1)
plt.title('Original image')
plt.imshow(HeadCT, cmap='gray', vmin=0, vmax=255)
plt.axis(False)

plt.subplot(1, 2, 2)
plt.title('Transformed image')
plt.imshow(t_HeadCT, cmap='gray', vmin=0, vmax=255)
plt.axis(False)

# PART D
plt.figure()
plt.title('Transform function')
# we determine the suitable coordinates for plotting the transform functions with step function
x = np.array([0, 10, 60, 255])
y = np.array([25, 25, 125, 25])
# set limits for both axes to make the scale values proportional to the 8-bit image intensity
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.step(x, y)  # plot the step function with the help of previously set coordinates

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
