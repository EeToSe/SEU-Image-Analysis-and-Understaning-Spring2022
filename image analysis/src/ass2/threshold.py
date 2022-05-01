import numpy as np
import matplotlib.pyplot as plt
import cv2

from otsu import *
from fuzzy_threshold import fuzzy_threshold

if __name__ == '__main__':
    file_loc = "../../data/ass2/"
    frame = np.loadtxt(file_loc+"squares.txt",delimiter=",").astype(np.uint8)

    ########
    # Otsu #
    range_th = range(256)
    # calculate the between-class variance
    sigma_square= [otsu_criteria_compute(frame, th) for th in range(256)]
    otsu_t = range_th[np.argmax(sigma_square)] 
    
    # threshold based the otsu result
    otsu_res = np.zeros(frame.shape, dtype=np.uint8)
    otsu_res[frame >= otsu_t] = 255

    # plot 
    x = np.arange(0, 256) 
    plt.figure(0)
    plt.title("$\sigma^{2}(t)$ as a function of $t$ ") 
    plt.xlabel("$t$") 
    plt.ylabel("$\sigma^{2}(t)$") 
    plt.plot(x,np.asarray(sigma_square)) 
    
    # save results
    plt.savefig('between-class variance.png')    
    cv2.imwrite("../../result/ass2/otsu.png", otsu_res)

    ################
    # FUZZY METHOD #
    S_t, fuzzy_t = fuzzy_threshold(frame)
    fuzzy_res = np.zeros(frame.shape, dtype=np.uint8)
    fuzzy_res[frame >= fuzzy_t] = 255

    # plot 
    plt.figure(1)
    plt.title("$S(t)$ as a function of $t$ ") 
    plt.xlabel("$t$") 
    plt.ylabel("$S(t)$") 
    plt.plot(x,S_t)

    # save results
    plt.savefig('Entropy.png') 
    cv2.imwrite("../../result/ass2/fuzzy.png", fuzzy_res)    

    