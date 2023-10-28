import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Input_data/noisy_image.png', 0)

img_ft = np.fft.fft2(img)  # Fourier transform
img_ftshift = np.fft.fftshift(img_ft)  # center the zero-frequency component
img_abs = np.log(np.abs(img_ftshift)) # magnitude spectrum

x, y = np.indices(img_ftshift.shape)
r, c = img_ftshift.shape
dis = (x - r // 2) ** 2 + (y - c // 2) ** 2
# img[(dis < 101**2) ^ (dis < 100**2)] = 0.5 * img.size
img_ftshift[(dis < 151 ** 2) ^ (dis < 150 ** 2)] = 0.2 * img_ftshift.size  # suitable mask for removing noise
# img[(dis < 201**2) ^ (dis < 200**2)] = 0.4 * img.size
# img[(dis < 251**2) ^ (dis < 250**2)] = 0.8 * img.size

img_ift = np.fft.ifftshift(img_ftshift)  # shift zero-frequency component to the initial position
img_iftshift = np.fft.ifft2(
    img_ift).real  # inverse Fourier transform and choosing the real part of the image for plotting

# plt.subplot(1, 2, 1)
# plt.title('Original noisy image')
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.axis(False)
#
# plt.subplot(1, 2, 2)
# plt.title('Filtered image')
# plt.imshow(img_iftshift, cmap='gray', vmin=0, vmax=255)
# plt.axis(False)
# plt.show()
