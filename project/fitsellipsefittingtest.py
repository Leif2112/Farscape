# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:25:22 2022

@author: Leif Tinwell
"""

# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np 

import math
import os
from os import listdir
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.timeseries import TimeSeries


# #read input
im = cv.imread('D:/Desktop/Uni/project/corrected/hmi.ic_45s.20220202_000000_TAI.2.continuum.fits.png')
#convert to grayscale
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# threshold
thresh = cv.threshold(imgray, 75 , 255, cv.THRESH_BINARY_INV)[1]
# Morphological Closing: Get rid of the hole
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
# Morphological opening: Get rid of the stuff at the top of the circle
thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
# find largest contour
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv.contourArea)
# fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
ellipse = cv.fitEllipse(big_contour)
(xc,yc),(d1,d2),angle = ellipse
print(xc,yc,d1,d2,angle)

# draw ellipse
result = im.copy()
cv.ellipse(result, ellipse, (0, 255, 0), 1)

# draw circle at center
xc, yc = ellipse[0]
cv.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)

# draw vertical line
# compute major radius
rmajor = max(d1,d2)/2
if angle > 90:
    angle = angle - 90
else:
    angle = angle + 90
print(angle)
xtop = xc + math.cos(math.radians(angle))*rmajor
ytop = yc + math.sin(math.radians(angle))*rmajor
xbot = xc + math.cos(math.radians(angle+180))*rmajor
ybot = yc + math.sin(math.radians(angle+180))*rmajor
cv.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 1)

cv.imwrite("fitted_ellipse.jpg", result)

cv.imshow("sunspot_thresh", thresh)
cv.imshow("umbra_ellipse", result)
cv.waitKey(0)
cv.destroyAllWindows()


#create threshold values for umbral and penumbral regions


#plot angle for each iteration, create histogram 
out_dir = 'downloads'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
fits_dir = os.fsdecode('D:/Desktop/Uni/project/downloads/')
directory = 'downloads'

























# def ellipse_fitting():
#     for filename in os.listdir(directory):
#         file = os.path.join(directory, filename)
#         fits_file = fits.open(file)
#         im = fits_file[1].data
#         imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#         blur = cv.GaussianBlur(imgray, (25,25), 0)
#         threshold = cv.threshold(blur, 195, 255, cv.THRESH_BINARY_INV)[1]
#         morph1 = cv.morphologyEx(threshold, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
#         morph2 = cv.morphologyEx(morph1, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (36, 36)))
#         contours = cv.findContours(morph2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#         contours = contours[0] if len(contours) == 2 else contours[1]
#         big_contour = max(contours, key=cv.contourArea)
#         ellipse = cv.fitEllipse(big_contour)
#         (xc,yc),(d1,d2),angle = ellipse
#         result = im.copy()
#         cv.ellipse(result, ellipse, (0, 255, 0), 1)
#         xc, yc = ellipse[0]
#         cv.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)
#         # rmajor = max(d1,d2)/2
#         # if angle > 90:
#         #     angle = angle - 90
#         # else:
#         #     angle = angle + 90
#         print(angle)
#         # xtop = xc + math.cos(math.radians(angle))*rmajor
#         # ytop = yc + math.sin(math.radians(angle))*rmajor
#         # xbot = xc + math.cos(math.radians(angle+180))*rmajor
#         # ybot = yc + math.sin(math.radians(angle+180))*rmajor
#         # cv.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 1)
#         cv.imshow("sunspot_thresh", threshold)
#         cv.imshow("umbra_ellipse", result)
#         cv.waitKey(0)
#         cv.destroyAllWindows()




# angle_data = list(range(1, 2880))
# time_data = list(range(1, 2880))

# def ellipse_fitting():
#     for filename in os.listdir(directory):
#         file = os.path.join(directory, filename)
#         im = cv.imread(file)
#         imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#         threshold = cv.threshold(imgray, 75, 255, cv.THRESH_BINARY_INV)[1]
#         morph1 = cv.morphologyEx(threshold, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
#         morph2 = cv.morphologyEx(morph1, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
#         contours = cv.findContours(morph2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#         contours = contours[0] if len(contours) == 2 else contours[1]
#         big_contour = max(contours, key=cv.contourArea)
#         ellipse = cv.fitEllipse(big_contour)
#         (xc,yc),(d1,d2),angle = ellipse
#         result = im.copy()
#         cv.ellipse(result, ellipse, (0, 255, 0), 1)
#         xc, yc = ellipse[0]
#         cv.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)
#         rmajor = max(d1,d2)/2
#         angle_data.append(angle)
#         # cv.imwrite('ellipsed_' + str(filename), result)

# ellipse_fitting()
# print(angle_data)
# minorLocator = MultipleLocator(60)

# fig, ax = plt.subplots(constrained_layout = True)
    
# ax.set(title='Sunspot Rotation Profile')
# ax.plot(angle_data, time_data, color='lime', label='Rotation')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
# ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
# ax.set_xlabel('Time')
# ax.set_ylabel('Angle (θ)')
# ax.grid(False)
# ax.legend(loc='upper left');
# plt.show()




 




