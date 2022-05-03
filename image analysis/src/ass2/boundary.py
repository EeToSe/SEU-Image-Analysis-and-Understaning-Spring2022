import numpy as np
import matplotlib.pyplot as plt
import cv2

from membership import *

# Membership function of input
d_range = range(-100,101)
mu_zero = [truncated_gaussian(d) for d in d_range]

# Membership function of the output
z_range = range(0, 256)
mu_black = [mu_black_function(z) for z in z_range]
mu_white = [mu_white_function(z) for z in z_range]

# dtype = float64 to deal with distance
frame = np.loadtxt("../../data/ass2/men.txt",delimiter=",").astype(np.float64)

res = frame.copy()
h, w = res.shape
for m in range(1, h-1):
  for n in range(1, w-1):
      neighbours = frame[m-1:m+2, n-1:n+2]
      res[m][n] = fuzzy_filtering(neighbours) 

test=np.load(file_loc+'edge.npy',encoding = "latin1").astype(np.uint8)
print(np.array_equal(test, res))

# plots of membership functions
plt.figure(0)
plt.title("$\mu_{zero}(d)$ as a function of $d$ ") 
plt.xlabel("$d$") 
plt.ylabel("$\mu_{zero}(d)$") 
plt.plot(d_range,mu_zero)
plt.waitforbuttonpress()

plt.figure(1)
plt.title("$\mu_{black}(z_{5})$ and $\mu_{white}(z_{5})$ as a function of $z_{5}$ ") 
plt.xlabel("$z_{5}$") 
plt.ylabel("$\mu_{black}(z_{5})$/ $\mu_{white}(z_{5})$") 
plt.plot(z_range,mu_black)
plt.plot(z_range,mu_white)
plt.legend(['$\mu_{black}(z_{5})$','$\mu_{white}(z_{5})$'])
plt.waitforbuttonpress()

plt.figure(2)
plt.plot(res, cmap='gray')
