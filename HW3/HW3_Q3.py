import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# PART A
img = cv.imread('Input_data/retina.jpg', 0)
h, w = img.shape
img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REFLECT_101)
median = cv.medianBlur(img, 3)
kernel = np.ones((3, 3), np.float32) / 9  # the required mask for averaging
average = cv.filter2D(img, -1, kernel).astype('uint8')

plt.figure()

plt.subplot(1, 2, 1)
plt.title('Median filter')
plt.imshow(median, cmap='gray')
plt.axis(False)

plt.subplot(1, 2, 2)
plt.title('Average filter')
plt.imshow(average, cmap='gray')
plt.axis(False)

plt.tight_layout()
plt.show()


# PART B
def pltransform(image, gamma):
    image = np.uint8(image)
    L = 255
    c = (L - 1) ** (1 - gamma)
    s = c * (image ** gamma)
    s = np.uint8(s)
    return s


# PART C
median_t = pltransform(median, 2 / 3)
# cv.imshow('Median transform', median_t)
plt.imshow(median_t, cmap='gray', vmin=0, vmax=255)
plt.title('Gamma transformation with gamma=2/3 on median filtered image')
plt.axis(False)
plt.show()

# PART D
lapkernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
mask = cv.filter2D(median_t, -1, lapkernel)
laptransform = pltransform(mask, 1 / 3)
# cv.imshow('Laplace transform', laptransform)
plt.imshow(laptransform, cmap='gray', vmin=0, vmax=255)
plt.title('Gamma transformation with gamma=1/3 on laplacian filtered image')
plt.axis(False)
plt.show()

# PART E
sequence = np.arange(-2,2, 0.01)  # arithmetic progression
fourcc = cv.VideoWriter_fourcc(*'MPEG')
vid = cv.VideoWriter('Retinavid.avi', fourcc, 20.0, (w, h))
median_t = np.float32(median_t)
mask = np.float32(mask)

for i in range(len(sequence)):
    frame = median_t + (sequence[i] * mask)
    frame[np.where(frame < 0)] = 0
    frame[np.where(frame > 255)] = 0  # cutting out the parts that are out of the required intensity boundaries
    frame = np.uint8(frame)
    cv.imshow('Retina', frame)
    if cv.waitKey(10) & 0xff == ord('q'):
        break

    vid.write(frame)
    vid.release()
cv.destroyAllWindows()
