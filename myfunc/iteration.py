import numpy as np
import cv2 as cv
from cv2 import BORDER_REFLECT

def iteration(I_start,se1,Is):
    I_finish = cv.bitwise_and(cv.dilate(I_start, se1, iterations=1), Is)
    if np.array_equal(I_start, I_finish):
        return I_finish
    else:
        return iteration(I_finish, se1, Is)

def convex_iteration(I_start,B1,B2):
    e1 = cv.erode(I_start,B1,iterations=1,borderType = BORDER_REFLECT)
    e2 = cv.erode(cv.bitwise_not(I_start),B2, iterations=1)
    I_finish = cv.bitwise_or((cv.bitwise_and(e1,e2)),I_start)
    if np.array_equal(I_start, I_finish):
        return I_finish
    else:
        return convex_iteration(I_finish, B1, B2)

def mr_iteration(I_start, B, I0):
    I_finish = cv.min((cv.dilate(I_start, B)), I0)
    if np.array_equal(I_start, I_finish):
        return I_finish
    else:
        return mr_iteration(I_finish, B, I0)