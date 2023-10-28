import cv2 as cv
import numpy as np

img_num = 2  # for MRI2 image's name in displaying
coordinates1 = []  # empty lists that obtain the user's chosen coordinates
coordinates2 = []


def choose_points(image, coordinates):
    cv.imshow(f"MRI {img_num}", image)
    print('Click on 3 coordinates and press Enter')

    def user_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            print(x, ' ', y)
            coordinates.append((x, y))
            cv.putText(image, str(x) + ',' + str(y), (x, y), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            cv.circle(image, (x, y), 5, (255, 0, 127), -1)  # show the chosen coordinates
            cv.imshow(f"MRI {img_num}", image)

    while True:
        cv.setMouseCallback(f"MRI {img_num}", user_click)
        k = cv.waitKey(0)
        if k == ord('\r'):
            break
            # end the function when user presses Enter key


img2 = cv.imread('Input_data/MRI2.jpg')
img2_copy = img2.copy()

img1 = cv.imread('Input_data/MRI.jpg')
img1_copy = img1.copy()

choose_points(img2_copy, coordinates2)

img_num = ''  # for MRI image's name in displaying
choose_points(img1_copy, coordinates1)

coordinates1 = np.float32(coordinates1)
coordinates2 = np.float32(coordinates2)

affine_matrix = cv.getAffineTransform(coordinates1, coordinates2)
output = cv.warpAffine(img1, affine_matrix, (img2.shape[1], img2.shape[0]))

cv.imshow('Output', output)
cv.waitKey(0)
cv.destroyAllWindows()
