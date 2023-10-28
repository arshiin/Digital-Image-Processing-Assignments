import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Input_data/CBC.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_blur = cv.medianBlur(img_gray, 5)  # to remove noise
img_edge = cv.Canny(img_blur, 15, 30)  # edge detection with canny method

# plt.imshow(img_edge, cmap='gray')
# plt.axis(False)
# plt.show()

circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, 220, param1=30, param2=15, minRadius=0,
                          maxRadius=115)
if circles is not None:
    circles = np.uint16(np.around(circles))  # convert circles output parameters (a, b, r) to integers

    for i in circles[0, :]:
        a, b, r = i[0], i[1], i[2]
        cv.circle(img, (a, b), r, (255, 0, 255), 5)  # circles circumferences with center a,b and radius r
        cv.circle(img, (a, b), 5, (0, 0, 255), -1)  # circles centers with radius 5

cv.namedWindow('Detected Circles', cv.WINDOW_NORMAL)
cv.imshow('Detected Circles', img)
cv.waitKey(0)
cv.destroyAllWindows()
