import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# PART A
def bitplane_slice(image, n):
    imlist = []
    h, w = image.shape
    if n in range(1, 9):
        for i in range(h):
            for j in range(w):
                imlist.append(
                    np.binary_repr(image[i][j],
                                   width=8))  # we add the string of 8-digit binary value of each pixel to the list
        # for each string of binary number (pixel), we take out the nth bit and multiply it by its value
        sliced = (np.array([i[8 - n] for i in imlist], dtype=np.uint8) * (2 ** (n - 1))).reshape(h, w)
        return sliced


img = cv.imread('Input_data/PCB.tif', 0)

plt.subplot(2, 4, 1)
plt.title('8th bit plane (MSB)')
plt.imshow(bitplane_slice(img, 8), cmap='gray')
plt.axis(False)

plt.subplot(2, 4, 2)
plt.title('7th bit plane')
plt.imshow(bitplane_slice(img, 7), cmap='gray')
plt.axis(False)

plt.subplot(2, 4, 3)
plt.title('6th bit plane')
plt.imshow(bitplane_slice(img, 6), cmap='gray')
plt.axis(False)

plt.subplot(2, 4, 4)
plt.title('5th bit plane')
plt.imshow(bitplane_slice(img, 5), cmap='gray')
plt.axis(False)

plt.subplot(2, 4, 5)
plt.title('4th bit plane')
plt.imshow(bitplane_slice(img, 4), cmap='gray')
plt.axis(False)

plt.subplot(2, 4, 6)
plt.title('3rd bit plane')
plt.imshow(bitplane_slice(img, 3), cmap='gray')
plt.axis(False)

plt.subplot(2, 4, 7)
plt.title('2nd bit plane')
plt.imshow(bitplane_slice(img, 2), cmap='gray')
plt.axis(False)

plt.subplot(2, 4, 8)
plt.title('1st bit plane (LSB)')
plt.imshow(bitplane_slice(img, 1), cmap='gray')
plt.axis(False)
plt.tight_layout()
plt.show()

# PART C
A = cv.imread('Input_data/NASA-A.tif', 0)
B = cv.imread('Input_data/NASA-B.tif', 0)
C = cv.imread('Input_data/NASA-C.tif', 0)

compare1 = compare2 = np.zeros((A.shape[0], A.shape[1]))  # These are the empty arrays to put our output images in
for i in range(5, 9):
    compare1 += ((bitplane_slice(A, i) ^ bitplane_slice(B, i)) * (2 ** i))  # comparison of NASA-A and NASA-B
    compare2 += ((bitplane_slice(B, i) ^ bitplane_slice(C, i)) * (2 ** i))  # comparison of NASA-B and NASA-C

plt.figure()
plt.subplot(1, 2, 1)
plt.title('A and B comparison')
plt.imshow(compare1, cmap='gray', vmin=0, vmax=255)
plt.axis(False)

plt.subplot(1, 2, 2)
plt.title('B and C comparison')
plt.imshow(compare2, cmap='gray', vmin=0, vmax=255)
plt.axis(False)
plt.show()
