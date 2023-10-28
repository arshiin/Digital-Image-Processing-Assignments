import cv2 as cv
import numpy as np

# PART A
img = cv.imread('Input_data/HeadCT.tif', 0)

ret, img_bin = cv.threshold(img, 100, 255, cv.THRESH_BINARY)

# PART B
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
img_closing = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)


# PART C
def hole_filling(image, seed):
    hf_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # kernel for filling the holes
    comp = 255 - image  # negative/compliment of the image
    h, w = image.shape
    img_x_1 = np.zeros((h, w), np.uint8)  # x(k-1)
    img_x = img_x_1.copy()  # x(k)
    img_x_1[seed] = 255  # this is the initial point that we start the hole filling with
    while np.any(img_x_1 != img_x):
        img_x = img_x_1
        img_x_1 = cv.dilate(img_x_1, hf_kernel)
        img_x_1 = img_x_1 & comp

    res = img_x | image  # add the main image to the final filled image to obtain the right result inside the boundary
    return res


# cv.imshow('Hole Filling', hole_filling(img_closing, (150, 300)))
# cv.waitKey(0)

# PART D
edge_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
dilation = cv.dilate(img_closing, edge_kernel)
erosion = cv.erode(img_closing, edge_kernel)
img_edge = dilation - erosion
