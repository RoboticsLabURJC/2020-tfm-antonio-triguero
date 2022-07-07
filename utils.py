import cv2 as cv
from cv2 import getStructuringElement
from cv2 import erode
import numpy as np

def skeleton(img):
    img = cv.inRange(img, (41, 41, 218), (41, 41, 218))

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    _, img = cv.threshold(img, 127, 255, 0)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False
    
    while ( not done):
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()
    
        zeros = size - cv.countNonZero(img)
        if zeros==size:
            done = True

    kernel = np.ones((3, 3))
    skel = cv.dilate(skel, kernel)
    return skel