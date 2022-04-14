#!/usr/bin/env python3
import cv2 as cv
from helper_function import read_txt, iteration,convex_iteration
import numpy as np

##################################################################
##################### Opening and Closing ########################
##################################################################

# load image txt file and save the np array as png
I0 = read_txt("../resource/ass1/butterfly.txt")
se1 = np.loadtxt("../resource/ass1/se_1.txt", delimiter=",").astype(np.uint8)

# I0 opening by se_1
I0_opening = cv.morphologyEx(I0, cv.MORPH_OPEN, se1)

# Complement of I0 opening by se_1
I0_comp = cv.bitwise_not(I0)
se1_hat = se1
I0_comp_opening = cv.morphologyEx(I0_comp, cv.MORPH_OPEN, se1_hat)

# I0 closing by B and take the complement of the result
I0_closing = cv.morphologyEx(I0, cv.MORPH_CLOSE, se1)
I0_closing_comp = cv.bitwise_not(I0_closing)

I0_opening_test = cv.imread("../img/Mor/I0_opening.png",0)
# Save results as png
cv.imwrite('../img/Mor/butterfly.png',I0)
cv.imwrite('../img/Mor/se_1.png',se1)
cv.imwrite('../img/Mor/I0_opening.png', I0_opening)
cv.imwrite('../img/Mor/I0_complement.png', I0_comp)
cv.imwrite('../img/Mor/I0_complement_opening.png', I0_comp_opening)
cv.imwrite('../img/Mor/I0_closing.png', I0_closing)
cv.imwrite('../img/Mor/I0_closing_complement.png', I0_closing_comp)

# Compare I0_closing_comp and I0_comp_opening
print(np.array_equal(I0_closing_comp, I0_comp_opening))

# ##################################################################
# ############### Morphological Reconstruction #####################
# ##################################################################
# # load the image of marker
Im = read_txt("../resource/ass1/marker.txt")
cv.imwrite("../img/marker.png", Im)
Is = I0_comp
I1 = cv.bitwise_and(cv.dilate(Im, se1, iterations=1), Is)
I2 = cv.bitwise_and(cv.dilate(I1, se1, iterations=1), Is)
I3 = cv.bitwise_and(cv.dilate(I2, se1, iterations=1), Is)
I4 = cv.bitwise_and(cv.dilate(I3, se1, iterations=1), Is)
I5 = cv.bitwise_and(cv.dilate(I4, se1, iterations=1), Is)

Ir_infinite = iteration(I5,se1,Is)

cv.imwrite("../img/Ir1.png", I1)
cv.imwrite("../img/Ir5.png", I5)
cv.imwrite("../img/Ir_infinite.png", Ir_infinite)
#
# ##################################################################
# ########################### Convex Hull ##########################
# ##################################################################
B11 = read_txt("../resource/ass1/convex_hull_se/B11.txt")
B12 = read_txt("../resource/ass1/convex_hull_se/B12.txt")
B21 = read_txt("../resource/ass1/convex_hull_se/B21.txt")
B22 = read_txt("../resource/ass1/convex_hull_se/B22.txt")
B31 = read_txt("../resource/ass1/convex_hull_se/B31.txt")
B32 = read_txt("../resource/ass1/convex_hull_se/B32.txt")
B41 = read_txt("../resource/ass1/convex_hull_se/B41.txt")
B42 = read_txt("../resource/ass1/convex_hull_se/B42.txt")


I_ch_0 = Ir_infinite
print(np.array_equal(cv.bitwise_not(I_ch_0),255-I_ch_0))
# Compute I_ch_i_infi
I_ch_1_infi =  convex_iteration(I_ch_0, B11, B12)
I_ch_2_infi =  convex_iteration(I_ch_0, B21, B22)
I_ch_3_infi =  convex_iteration(I_ch_0, B31, B32)
I_ch_4_infi =  convex_iteration(I_ch_0, B41, B42)
I_ch12 = cv.bitwise_or(I_ch_1_infi,I_ch_2_infi)
I_ch34 = cv.bitwise_or(I_ch_3_infi,I_ch_4_infi)
I_ch = cv.bitwise_or(I_ch12,I_ch34)

cv.imwrite("../img/I_ch_1_infi.png",I_ch_1_infi)
cv.imwrite("../img/I_ch_2_infi.png",I_ch_2_infi)
cv.imwrite("../img/I_ch_3_infi.png",I_ch_3_infi)
cv.imwrite("../img/I_ch_4_infi.png",I_ch_4_infi)
cv.imwrite("../img/I_ch.png",I_ch)

# Check if I_ch is the convex hull
print(np.array_equal(cv.bitwise_or(I_ch,I_ch_0), I_ch))
