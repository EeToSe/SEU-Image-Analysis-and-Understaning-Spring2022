import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


if __name__ == "__main__":
    image1 = np.loadtxt('img_1.txt', delimiter=",").astype(np.uint8)
    image2 = np.loadtxt('img_2.txt', delimiter=",").astype(np.uint8)

    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
    
    # find homography using match points with RANSAC
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = image1.shape
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    # apply perspective transform to image1
    image1_proj = cv.warpPerspective(image1,M,(h,w))
    print('Perspective Transform Matrix:\n {}'.format(M))
    image_stitch = np.maximum(image1_proj, image2)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    image_match = cv.drawMatches(image1,kp1,image2,kp2,good,None,**draw_params)
    
    plt.imshow(image1_proj, cmap='gray')
    plt.show()
    plt.imshow(image_stitch, cmap='gray')
    plt.show()
    plt.imsave('./stitching.png', image_stitch, cmap = 'gray')
    plt.imsave('./matches.png', image_match, cmap = 'gray')