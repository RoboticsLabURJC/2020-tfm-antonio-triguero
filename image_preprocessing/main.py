import cv2 as cv
import numpy as np
import utils

img = cv.imread('./img.png')
img = utils.skeleton(img)

obs_space_n = 8
bins = np.arange(0, img.shape[1], img.shape[1] // obs_space_n)
points = img[img.shape[0] // 2 + 5], img[img.shape[0] // 4 * 3], img[img.shape[0] - 5]
points = np.argwhere(points)

points_to_bins = [np.argwhere([bins[i] <= num <= bins[i+1] for i in range(len(bins) - 1)])[0, 0] for num in points[:, 1]]

print(int(np.mean(points_to_bins)))

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows() 