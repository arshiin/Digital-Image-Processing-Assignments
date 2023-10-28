import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((800,800))

x, y = np.indices(img.shape)
r, c = img.shape
dis = (x-r//2)**2+(y-c//2)**2
img[(dis < 101**2) ^ (dis < 100**2)] = 0.5 * img.size
img[(dis < 151**2) ^ (dis < 150**2)] = 0.2 * img.size
img[(dis < 201**2) ^ (dis < 200**2)] = 0.4 * img.size
img[(dis < 251**2) ^ (dis < 250**2)] = 0.8 * img.size

plt.figure()
plt.imshow(img, cmap='gray')
plt.show()