import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def blend_mask(im1, im2, homography1, homography2, out_size):
    """ This is to generate the warped masks for the input images
        for further image blending.
    Args:
      im1: input image1 in numpy.array with size [H, W, 3]
      im2: input image2 in numpy.array with size [H, W, 3]
      homography1: the homography to warp image1 onto the panorama
      homography2: the homography to warp image2 onto the panorame
      out_shape: the size of the final panarama, in format of (width, height)
    Returns:
      warp_mask1: The warped mask for im1, namely the weights for im1 to blend
      warp_mask2: The warped mask for im2, namely the weights for im2 to blend
    """
    im1H, im1W, _ = im1.shape
    im2H, im2W, _ = im2.shape

    # create mask for im1, zero at the borders and 1 at the center of the image
    mask1 = np.zeros((im1H, im1W))
    mask1[0, :] = 1
    mask1[-1, :] = 1
    mask1[:, 0] = 1
    mask1[:, -1] = 1
    mask1 = distance_transform_edt(1 - mask1)
    mask1 = mask1 / np.max(mask1)
    warp_mask1 = cv2.warpPerspective(mask1, homography1, out_size)

    # create mask for im2, zero at the borders and 1 at the center of the image
    mask2 = np.zeros((im2H, im2W))
    mask2[0, :] = 1
    mask2[-1, :] = 1
    mask2[:, 0] = 1
    mask2[:, -1] = 1
    mask2 = distance_transform_edt(1 - mask2)
    mask2 = mask2 / np.max(mask2)
    warp_mask2 = cv2.warpPerspective(mask2, homography2, out_size)

    # combine mask1 and mask2 to calculate the weights for im1 and im2 for blending.
    sum_mask = warp_mask1 + warp_mask2
    warp_mask1 = warp_mask1 / sum_mask
    warp_mask1 = np.stack((warp_mask1, warp_mask1, warp_mask1), axis=2)
    return warp_mask1

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix

    Inputs             Description
    ---------------------------------------------------------
    img1&2             two images to be stitched        
    H2to1              homography to warp image 2 to image 1's frame    
    
    Output             Description
    ---------------------------------------------------------
    imPano             Blends img1 and warped img2 and outputs the panorama image
    '''
    imH1, imW1, _ = im1.shape
    imH2, imW2, _ = im2.shape
    # warp corner points to find out_size
    corners = np.array([[0, 0, imW2-1, imW2-1], [0, imH2-1, 0, imH2-1], [1, 1, 1, 1]])
    cornersPano = np.matmul(H2to1, corners)
    cornersPano /= cornersPano[2, :]
    imW = np.int16(np.round(np.max(cornersPano[0,:])))
    imH = np.int16(np.round(np.max(cornersPano[1,:])))
    # warp im2 into the im1's reference
    im2_project = cv2.warpPerspective(im2, H2to1, (imW, imH))
    # cv2.imwrite('../results/im2to1.png', im2_project)

    # simply overlay two images with seams at the edges
    imPano = np.zeros((imH, imW, 3), np.uint8)
    imPano[:imH1, :imW1, :] = im1
    im2Pixels = np.where(imPano==0)
    imPano[im2Pixels] = im2_project[im2Pixels]
    return imPano

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix

    Inputs             Description
    ---------------------------------------------------------
    img1&2             two images to be stitched        
    H2to1              homography to warp image 2 to image 1's frame    
    
    Output             Description
    ---------------------------------------------------------
    imPano             Blends img1 and warped img2 and outputs the panorama image
    '''
    imH1, imW1, _ = im1.shape
    imH2, imW2, _ = im2.shape
    
    # apply H2to1 to corner points to find M and out_size
    corners = np.array([[0, 0, imW2-1, imW2-1], [0, imH2-1, 0, imH2-1], [1, 1, 1, 1]])
    cornersPano = np.matmul(H2to1, corners)
    cornersPano /= cornersPano[2, :]
    x_min, y_min = cornersPano.min(axis=1)[:2]
    x_max, y_max = cornersPano.max(axis=1)[:2]
    tx = -x_min if x_min < 0 else 0 
    ty = -y_min if y_min < 0 else 0
    imW = np.int16(np.round(x_max+tx))
    imH = np.int16(np.round(y_max+ty))
    out_size = (np.max([imW1,imW]), np.max([imH1,imH]))
    M = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    
    # warp images in a common reference
    im1_warp = cv2.warpPerspective(im1, M, out_size)
    im2_warp = cv2.warpPerspective(im2, np.matmul(M,H2to1), out_size)
    
    # generate a mask for each image you wish to blend
    warp_mask1 = blend_mask(im1, im2, M, np.matmul(M, H2to1), out_size)
    warp_mask2 = 1- warp_mask1  
    imBlend = warp_mask1*im1_warp + warp_mask2*im2_warp
    return imBlend

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=500, tol=3)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    #plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/q6_1.npy', H2to1)

    pano_im_6_1 = imageStitching(im1, im2, H2to1)
    cv2.imwrite('../results/6_1.jpg', pano_im_6_1)

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)

    im3 = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', im3)
    cv2.imshow('panoramas', im3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
