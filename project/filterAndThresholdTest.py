# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:39:32 2022

@author: Leif Tinwell
"""
from mpl_toolkits import mplot3d 
import cv2  
import numpy as np 
from matplotlib import pyplot as plt 
import os 
import numpy as np
import matplotlib.pyplot as plt
import astropy.io
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import os
import sunpy.map as Map
import drms
import cv2 as cv
from PIL import Image



# file = os.path.join('D:/Desktop/Uni/project/AR2939/.FITS Data/hmi.ic_45s.20220205_224200_TAI.2.continuum.fits')
# image_data = fits.getdata(file, ext=1)
# fig = plt.figure(figsize=(8,8), frameon=False)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
# plt.imshow(image_data, cmap = 'gray',origin='lower')


img = cv2.imread('D:/Desktop/Uni/project/AR2939/Image Data/hmi.ic_45s.20220205_224200_TAI.2.continuum.fits.png')

img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Gaussian_Blur = cv2.GaussianBlur(img0, (45,45), 55) 
# Gaussian_Blur2 = cv.GaussianBlur(img0, (25,25), 0)

ret,th1 = cv2.threshold(Gaussian_Blur, 70, 255, cv2.THRESH_BINARY_INV)
ret,th1_5 = cv2.threshold(Gaussian_Blur, 90, 255, cv2.THRESH_BINARY_INV)
ret,th2 = cv2.threshold(Gaussian_Blur, 120, 255, cv2.THRESH_BINARY_INV)
# ret,th3 = cv2.threshold(Gaussian_Blur, 150, 255, cv2.THRESH_BINARY_INV)
ret,th4 = cv2.threshold(Gaussian_Blur, 215, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours1, hierarchy = cv2.findContours(th1_5,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# contours3, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours4, hierarchy = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img1 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) 

# cv2.drawContours(img1, contours, -1, (96,20,69), 2)
# cv2.drawContours(img1, contours1, -1, (114,81,51), 2)
cv2.drawContours(img1, contours2, -1, (127,129,57), 2)
# cv2.drawContours(img1, contours3, -1, (92,167,98), 2)
cv2.drawContours(img1, contours4, -1, (42,193,169), 2)
cnt = contours4[0]
cnt1 = contours2[0]
ellipse = cv.fitEllipse(cnt)
ellipse1 = cv.fitEllipse(cnt1)
# cv2.drawContours(img1, contours2, -1, (127,129,57), 2)
im = cv.ellipse(img1,ellipse,0,2)
im = cv.ellipse(img1,ellipse1,0,2)
(xc,yc),(d1,d2),angleu = ellipse
xc, yc = ellipse[0]
# cv2.circle(im, (int(xc),int(yc)), 1, (42,193,169), -1)
cv2.imshow('test', im)
cv2.waitKey(0)
cv2.destroyAllWindows()








'''
#451460
#335172
#39817f
#63c15c
#a9c12a
'''


# cnts, hier = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw found contours in input image
# image = cv2.drawContours(img0, cnts, -1, (0, 0, 255), 2)

# for i, cont in enumerate(cnts):
#     h = hier[0, i, :]
#     print(h)
#     if h[3] != -1:
#         elps = cv2.fitEllipse(cnts[i])
#         (xc,yc),(d1,d2),angle = elps
#     elif h[2] == -1:
#         elps = cv2.fitEllipse(cnts[i])
#         (xc,yc),(d1,d2),angle = elps
#         print(f'angle of {angle}')
#     cv2.ellipse(image, elps, (255, 0, 0), 2)

# # Downsize image
# cv2.imshow("Output image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# def penumbra_ellipse_fitting():
#     for filename in os.listdir('D:/Desktop/Uni/project/corrected/'):
#         file = os.path.join('D:/Desktop/Uni/project/corrected/', filename)
#         img0 = cv2.imread(file, 0)                      #collect files from directory
#         Gaussian_Blur = cv2.GaussianBlur(img0, (45,45), 55) 
#         ret,th4 = cv2.threshold(Gaussian_Blur, 40, 255, cv2.THRESH_BINARY_INV)
#         cnts, hier = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         image = cv2.drawContours(img0, cnts, -1, (0, 0, 255), 2)
#         for i, cont in enumerate(cnts):
#             h = hier[0, i, :]
#             print(h)
#             if h[3] != -1:
#                 elps = cv2.fitEllipse(cnts[i])
#                 (xc,yc),(d1,d2),angle = elps
#             elif h[2] == -1:
#                 elps = cv2.fitEllipse(cnts[i])
#                 (xc,yc),(d1,d2),angle = elps
#                 print(f'angle of {angle}')
#             cv2.ellipse(image, elps, (255, 0, 0), 2)
#             cv2.imshow("output", image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()


# umbra_angle_data = []
# def umbra_ellipse_fitting():
#     for filename in os.listdir('D:/Desktop/Uni/project/AR2939/Image Data/'):
#         file = os.path.join('D:/Desktop/Uni/project/AR2939/Image Data/', filename)
#         img0 = cv2.imread(file, 0)                      #collect files from directory
#         Gaussian_Blur = cv2.GaussianBlur(img0, (45,45), 25) 
#         ret,th4 = cv2.threshold(Gaussian_Blur, 215 ,255 ,cv2.THRESH_BINARY_INV)
#         cnts, hier = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         image = cv2.drawContours(img0, cnts, -1, (0, 0, 255), 2)
#         for i, cont in enumerate(cnts):
#             elps = cv2.fitEllipse(cnts[i])
#             (xc,yc),(d1,d2),angle = elps
#             umbra_angle_data.append(int(angle))
#             print(angle)
#             cv2.ellipse(image, elps, (255, 0, 0), 2)
#             xc, yc = elps[0]
#             cv2.circle(image, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot
#             cv2.imshow("output", image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
# area_val = []
# def umbra_ellipse_fitting():
#     for filename in os.listdir('D:/Desktop/Uni/project/corrected/'):
#         file = os.path.join('D:/Desktop/Uni/project/corrected/', filename)
#         im = cv2.imread(file)                      #collect files from directory
#         imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       #convert to grayscale
#         blur = cv2.GaussianBlur(imgray, (25,25), 25)
#         #threshold to eliminate excess data, focus on penumbra area of sunspot
#         threshold = cv2.threshold(blur, 215, 255, cv2.THRESH_BINARY_INV)[1] 
#         #find contours of threshold silhouette
#         contours = cv2.findContours(threshold, 
#                                     cv2.RETR_TREE, 
#                                     cv2.CHAIN_APPROX_NONE
#                                     )
#         contours = contours[0] if len(contours) == 2 else contours[1]
#         big_contour = max(contours, key=cv2.contourArea)
#         ellipse = cv2.fitEllipse(big_contour)       #fit ellipse around contours
#         image = cv2.drawContours(im, big_contour, -1, (0, 0, 255), 2)
#         (xc,yc),(d1,d2),angleu = ellipse
#         cv2.ellipse(image, ellipse, (0, 255, 0), 1)
#         xc, yc = ellipse[0]
#         cv2.circle(image, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot
#         cv2.imshow("output", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        

# umbra_ellipse_fitting()
# print(umbra_angle_data)
# print(area_val)











# cnt = contours4[0]
# ellipse = cv.fitEllipse(cnt)


# cv.drawContours(img0, contours, -1, 0, 1)

# cv.drawContours(img0, contours2, -1, 0, 1)

# cv.drawContours(img0, contours3, -1, (255, 0 ,0 ), 1)







# blur = cv.GaussianBlur(img0, (45,45), 55)
#threshold to eliminate excess data, focus on penumbra area of sunspot
# threshold = cv.threshold(blur, 101, 23, cv.THRESH_BINARY)[1] 
#find contours of threshold silhouette
# contours, hierarchy = cv.findContours(threshold, 
                            # cv.RETR_TREE, 
                            # cv.CHAIN_APPROX_SIMPLE
                            # )
# contours = contours[0] if len(contours) == 2 else contours[1]
# big_contour = max(contours, key=cv.contourArea)
# cv.drawContours(img0, contours, -1, 255, 1)
# ellipse = cv.fitEllipse(big_contour)       #fit ellipse around contours
# (xc,yc),(d1,d2),angleu = ellipse
# result = img0.copy()
# cv.ellipse(result, ellipse, (0, 255, 0), 1)
# xc, yc = ellipse[0]
# cv.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot

  
# titles = ['Image',
#           'Gaussian BLur', 'Otsu threshold',
#             '.']

# images = [img0, Gaussian_Blur, th1, None]

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot(img0, th1)
# plt.show()


# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

