import cv2 as cv
import numpy as np

# PART A AND B
img1 = cv.imread('Input_data/MRI-Head.png', 0).astype(np.int32)
vid = cv.VideoCapture('MRI.avi')
ret, first_frame = vid.read()
first_frame_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY).astype(np.int32)
noise = cv.absdiff(img1, first_frame_gray) # Subtracting the images to get noise

# Find the average of every element in our noise array
avg = np.average(noise)
std = np.std(noise)
print('\nAverage:', avg, '\nStandard Deviation:', std, sep=' ')
# noise = np.uint8(noise)
# cv.imshow('noise',noise)

# We put every frame of the video in a list to calculate the average frame later
vid = cv.VideoCapture('Input_data/MRI.avi')
frames = []
i = 0
frame_numbers = int(vid.get(cv.CAP_PROP_FRAME_COUNT))  # to get the number of frames
for i in range(frame_numbers):
    ret, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frames.append(gray)
    i += 1

avg_frame = np.average(frames, axis=0).astype(np.uint8)  # axis=0 since it's a list and we want the rows of it
cv.imshow('Average Frame', avg_frame)
cv.imwrite('Average Frame.png', avg_frame)

# PART C
avg_frame = np.int32(avg_frame)
noise2 = cv.absdiff(img1, avg_frame)
avg2 = np.average(noise2)
std2 = np.std(noise2)
print('\nAverage 2:', avg2, '\nStandard Deviation 2:', std2, sep=' ')
# noise2 = np.uint8(noise2)
# cv.imshow('noise', noise2)

# Noise calculation
sigma = std
sigma2 = std2
k = (sigma ** 2) / (sigma2 ** 2)
print('\nK=', k)

# PART D
img3 = cv.imread('Average Frame.png', 0)
img4 = cv.imread('Input_data/mask.png', 0)

for i in range(img4.shape[0]):
    for j in range(img3.shape[1]):
        if img4[i, j] == 255:
            img4[i, j] = img3[i, j]

cv.imshow('Masked image', img4)

vid.release()
cv.waitKey(0)
cv.destroyAllWindows()
