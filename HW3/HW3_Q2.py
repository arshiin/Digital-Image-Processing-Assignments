import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# PART A
def func(image, filname, ksize):
    height, width = image.shape
    kernel = np.ones((ksize, ksize), np.float32)
    output = np.zeros((height, width), dtype='float')
    n = (ksize - 1) // 2
    image = cv.copyMakeBorder(img, n, n, n, n, cv.BORDER_REFLECT_101)
    if filname == 'averaging':
        mask = kernel / (ksize ** 2)
        for i in range(n, height - n):
            for j in range(n, width - n):
                for x in range(-n, n + 1):
                    for y in range(-n, n + 1):
                        output[i][j] += image[i + x, j + y] * mask[x + n, y + n]
        return output

    elif filname == str('minimum'):
        minimg = np.zeros(ksize ** 2)
        for i in range(n, height - n):
            for j in range(n, width - n):
                c = 0
                for x in range(-n, n + 1):
                    for y in range(-n, n + 1):
                        minimg[c] = image[i + x, j + y]
                        c += 1
                minimg = sorted(minimg)
                output[i, j] = minimg[0]
        return output

    elif filname == 'median':
        centerimg = np.zeros(ksize ** 2)
        for i in range(n, height - n):
            for j in range(n, width - n):
                c = 0
                for x in range(-n, n + 1):
                    for y in range(-n, n + 1):
                        centerimg[c] = image[i + x, j + y]
                        c += 1
                centerimg = sorted(centerimg)
                output[i, j] = centerimg[((ksize ** 2) - 1) // 2]
        return output

    elif filname == str('sobel_y'):
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                for x in range(-n, n + 1):
                    for y in range(-n, n + 1):
                        output[i, j] += (image[i + x, j + y] * sobel_y[x + n, y + n])
        return output

    elif filname == str('laplacian'):
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                for x in range(-n, n + 1):
                    for y in range(-n, n + 1):
                        output[i, j] += (image[i + x, j + y] * laplacian[x + n, y + n])
        return output

    # PART B
    elif filname == str('diagonal_filter'):
        mask = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
        for i in range(n, height - n):
            for j in range(n, width - n):
                for x in range(-n, n + 1):
                    for y in range(-n, n + 1):
                        output[i, j] += (image[i + x, j + y] * mask[x + n, y + n])
        return output


# PART C
img = cv.imread('Input_data/MRI.png', 0)
fig = plt.figure()

plt.subplot(3, 3, 1)
plt.title('Averaging (3)')
plt.imshow(func(img, 'averaging', 3), cmap='gray')

plt.subplot(3, 3, 2)
plt.title('Minimum (3)')
plt.imshow(func(img, 'minimum', 3), cmap='gray')

plt.subplot(3, 3, 3)
plt.title('Median (3)')
plt.imshow(func(img, 'median', 3), cmap='gray')

plt.subplot(3, 3, 4)
plt.title('Laplacian')
plt.imshow(func(img, 'laplacian', 3), cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 3, 5)
plt.title('Sobel_y')
plt.imshow(func(img, 'sobel_y', 3), cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 3, 6)
plt.title('"Diagonal" filter')
plt.imshow(func(img, 'diagonal_filter', 3), cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 3, 7)
plt.title('Averaging (7)')
plt.imshow(func(img, 'averaging', 7), cmap='gray')

plt.subplot(3, 3, 8)
plt.title('Minimum (7)')
plt.imshow(func(img, 'minimum', 7), cmap='gray')

plt.subplot(3, 3, 9)
plt.title('Median (7)')
plt.imshow(func(img, 'median', 7), cmap='gray')

for ax in fig.axes:
    ax.axis(False)
plt.tight_layout()
plt.show()
