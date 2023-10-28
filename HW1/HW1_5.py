import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# Part A
img = cv.imread('Input_data/AUT-DIP.png', 0)
height, width = img.shape
h = height // 2
w = width // 3
m = 1
n = 1

for i in range(0, img.shape[0], h):
    for j in range(0, img.shape[1], w):
        n += 1
        while m < n <= 7 and m <= 7:
            cv.imwrite(f'Part {m}.png', img[i:i + h, j:j + w])
            m += 1

# Part B
P1 = cv.imread('Part 1.png')
scaled = cv.resize(P1, None, None, fx=2, fy=2, interpolation=0)
h1, w1, c1 = P1.shape
hs1, ws1, cs1 = scaled.shape
hcrop = (hs1 - h1) // 2
wcrop = (ws1 - w1) // 2
cropped = scaled[hcrop:hcrop + h1, wcrop:wcrop + w1]
# cv.imshow('Part 1 Scaled', scaled)
# cv.imshow('Part 1 Cropped', cropped)

# PART C
P2 = cv.imread('Part 2.png')
SM = np.float32([[1, 0.2, 0], [0, 1, 0]])
sheared = cv.warpAffine(P2, SM, (500, 500))
# cv.imshow('Part 2 Horizontally Sheared', sheared)

# Part D
P3 = cv.imread('Part 3.png')
TM = np.float32([[1, 0, -80], [0, 1, 100]])
translated = cv.warpAffine(P3, TM, (500, 500))
# cv.imshow('Part 3 Translated', translated)

# PART E
P4 = cv.imread('Part 4.png')
h4, w4, c4 = P4.shape
FM = np.zeros((h4, w4, c4), dtype='uint8')
x = h4 / 2
y = w4 / 2
angle = np.radians(25)
for i in range(h4):
    for j in range(w4):
        xforward = int((i - x) * math.cos(angle) - (j - y) * math.sin(angle) + x)
        yforward = int((i - x) * math.sin(angle) + (j - y) * math.cos(angle) + x)
        if (xforward < h4) and (yforward < w4):
            FM[i, j] = P4[xforward, yforward]

# cv.imshow('Part 4 rotated with forward mapping', forward)

# PART F
P5 = cv.imread('Part 5.png')
h5, w5, c5 = P5.shape
BM = np.zeros((h5, w5, c5), dtype='uint8')
x = h5 / 2
y = w5 / 2
angle = np.radians(-25)
for i in range(h5):
    for j in range(w5):
        xbackward = int((i - x) * math.cos(angle) + (j - y) * math.sin(angle) + x)
        ybackward = int(-(i - x) * math.sin(angle) + (j - y) * math.cos(angle) + x)
        if (xbackward < h5) and (ybackward < w5):
            BM[i, j] = P5[xbackward, ybackward]
# cv.imshow('Part 5 rotated with backward mapping', backward)

# Part G
P6 = cv.imread('Part 6.png')
h6, w6, c6 = P6.shape
RM = cv.getRotationMatrix2D((w6 / 2, h6 / 2), 45, 1)
rotated = cv.warpAffine(P6, RM, (w6, h6))
# cv.imshow('Part 6 Rotated 45 Degrees', rotated)

# PART H
plt.figure(figsize=(6,4))
plt.suptitle('Images geometrical transformations')
plt.subplot(2, 3, 1)
plt.title('scaled & cropped')
plt.axis(False)
plt.imshow(cropped, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('horizontally sheared')
plt.axis(False)
plt.imshow(sheared, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('translated')
plt.axis(False)
plt.imshow(translated, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('rotated with\nforward mapping')
plt.axis(False)
plt.imshow(FM, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('rotated with\nbackward mapping')
plt.axis(False)
plt.imshow(BM, cmap='gray')

plt.subplot(2, 3, 6)
plt.title('rotated')
plt.axis(False)
plt.imshow(rotated, cmap='gray')
plt.tight_layout()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
