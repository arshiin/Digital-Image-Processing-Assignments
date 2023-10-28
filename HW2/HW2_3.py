import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# PARTS A AND B
def transform(img, bd):
    img = np.float64(img)
    L = 2 ** bd - 1
    # maximum value for a is 1/L so we rewrite s(r) function and substitute a
    img2 = (img ** 2 + img) / L
    transformed_img = np.round(img2)
    transformed_img = transformed_img.astype(img.dtype)  # to make the output type the same as input type
    return transformed_img, L


# PART C
kidney = cv.imread('Input_data/kidney.tif')
t_kidney, L_kidney = transform(kidney, 8)
t_kidney = np.uint8(t_kidney)

chest = cv.imread('Input_data/chest.tif', cv.IMREAD_ANYDEPTH)
t_chest, L_chest = transform(chest, 16)
t_chest = np.uint16(t_chest)

# Kidney plots
plt.figure()
plt.subplot(2, 2, 1)
plt.title('Original image')
plt.imshow(kidney, cmap='gray', vmin=0, vmax=L_kidney)
plt.axis(False)

plt.subplot(2, 2, 2)
plt.title('Transformed image')
plt.imshow(t_kidney, cmap='gray', vmin=0, vmax=L_kidney)
plt.axis(False)

plt.subplot(2, 2, 3)
plt.title('Original image histogram')
plt.hist(kidney.ravel(), bins=range(0, L_kidney, 4))

plt.subplot(2, 2, 4)
plt.title('Transformed image histogram')
plt.hist(t_kidney.ravel(), bins=range(0, L_kidney, 4))
plt.tight_layout()

# Chest plots
plt.figure()
plt.subplot(2, 2, 1)
plt.title('Original image')
plt.imshow(chest, cmap='gray', vmin=0, vmax=L_chest)
plt.axis(False)

plt.subplot(2, 2, 2)
plt.title('Transformed image')
plt.imshow(t_chest, cmap='gray', vmin=0, vmax=L_chest)
plt.axis(False)

plt.subplot(2, 2, 3)
plt.title('Original image histogram')
plt.hist(chest.ravel(), bins=range(0, L_chest, 1024))

plt.subplot(2, 2, 4)
plt.title('Transformed image histogram')
plt.hist(t_chest.ravel(), bins=range(0, L_chest, 1024))
plt.tight_layout()

# PART D
plt.figure()
plt.title('State space plots')
x = np.linspace(0, L_chest)  # create an array based on a 16-bit image to plot the functions
t = (x ** 2 + x) / L_chest  # repeating the s(r) function to plot
plt.plot(x, t, c='k', label='Transform function')
plt.plot(x, x, c='b', linestyle='--', label='Identify function')
plt.legend()

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
