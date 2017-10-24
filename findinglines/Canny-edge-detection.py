import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('/home/yuchen/Pictures/exit-ramp')
plt.imshow(image)

import cv2
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # graysc convers
plt.imshow(gray,cmap='gray')

# output edges is a binary img with white pixels tracing out the detected edges
# and black else, resonable thresholds 10s~1000s
# ratio of low to high : 1:2~1:3

#include gussian smoothing before running canny

#larger kernel_size--smoothing over a larger area


kernel_size =3
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
low_threshold =100
high_threshold =200
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.imshow(edges,cmap='Greys_r')