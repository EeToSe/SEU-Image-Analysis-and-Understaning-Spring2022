#!/usr/bin/env python3
import cv2 as cv
from helper_function import read_txt, mr_iteration
import numpy as np

# Top-hat transform
I0 = cv.imread("../resource/ass2/rice.png", cv.IMREAD_GRAYSCALE)
I0_th = I0.copy()
I0_th[I0_th <= 150] = 0
I0_th[I0_th > 150] = 255

# compute I_top-hat
se1 =  np.loadtxt("../resource/ass2/se_1.txt", delimiter=",").astype(np.uint8)
I0_opening = cv.morphologyEx(I0, cv.MORPH_OPEN, se1)
I_tophat = I0 - I0_opening

# compute I_top-hat-th
I_tophat_th = I_tophat.copy()
I_tophat_th[I_tophat_th <= 60] = 0
I_tophat_th[I_tophat_th > 60] = 255

# opening, remove the light point
se2 = np.loadtxt("../resource/ass2/se_2.txt", delimiter=",").astype(np.uint8)
I_remove = cv.morphologyEx(I_tophat_th, cv.MORPH_OPEN, se2)

cv.imwrite('../img/Gray/I0th.png',I0_th)
cv.imwrite('../img/Gray/I_tophat.png',I_tophat)
cv.imwrite('../img/Gray/I_tophat_th.png',I_tophat_th)
cv.imwrite('../img/Gray/I_remove.png',I_remove)

# H-dome
Im = I0.copy()
Im = np.int16(Im) # marker image
Im = Im - 45
B3 = np.loadtxt("../resource/ass2/se_3.txt", delimiter=",").astype(np.uint8)

Ir0 = Im
Ir1 = cv.min((cv.dilate(Ir0,B3)),np.int16(I0))
Ir2 = cv.min((cv.dilate(Ir1,B3)),np.int16(I0))
Ir3 = cv.min((cv.dilate(Ir2,B3)),np.int16(I0))
Ir4 = cv.min((cv.dilate(Ir3,B3)),np.int16(I0))
Ir5 = cv.min((cv.dilate(Ir4,B3)),np.int16(I0))
Ir_infi = mr_iteration(Ir0, B3, np.int16(I0))

cv.imwrite("../img/Gray/B3.png",B3)
cv.imwrite("../img/Gray/Imarker.png", Im)
cv.imwrite("../img/Gray/Ir1.png", Ir1)
cv.imwrite("../img/Gray/Ir5.png", Ir5)
cv.imwrite("../img/Gray/Ir_infi.png", Ir_infi)
cv.imwrite("../img/Gray/I0-Ir_infi.png", I0-Ir_infi)