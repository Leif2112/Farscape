# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:19:50 2022

@author: Leif Tinwell
"""
import os
from os import listdir
import drms
import math
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.dates as mdates
import astropy.time
from astropy.coordinates import SkyCoord
import astropy.units as u 
import sunpy.map 
from sunpy.net import Fido
from astropy.io import fits
from sunpy.net import jsoc, fido_factory, attrs as a
from sunpy.timeseries import TimeSeries
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import ImageNormalize, SqrtStretch
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
from astropy.visualization import astropy_mpl_style
from astropy.wcs import WCS
plt.style.use(astropy_mpl_style)

###############################################################################
#initializing DRMS client to given email address
client = drms.Client(email = 'leiftinwell@rocketmail.com', verbose=True)

out_dir = 'downloads'
if not os.path.exists(out_dir):                   #construct download directory
    os.mkdir(out_dir)
fits_dir = os.fsdecode('D:/Desktop/Uni/project/downloads/')
directory = 'downloads'
###############################################################################

# #reference time of sequence
t0 = astropy.time.Time('2022-02-05T13:22:08', scale='utc', format='isot')

#query full frame HMI image from fido for quick visualization of targeted sunspot
q = Fido.search(   
    a.Instrument.hmi,
    a.Physobs.intensity,
    a.Time(t0, t0 + 10*u.s)
    )
qf = Fido.fetch(q)
m = sunpy.map.Map(qf).rotate(order=3)  #rotate for Solar North pointing upwards

# create submap from this image, and crop to active region 
m_cutout = m.submap(
    SkyCoord(-32*u.arcsec, -142*u.arcsec, frame=m.coordinate_frame),
    top_right=SkyCoord(30*u.arcsec, -187*u.arcsec, frame=m.coordinate_frame)
    )

dl_map = sunpy.map.Map(m_cutout)
dl_map.plot()



###############################################################################
# The HMI continuum data refers to the map of the continuum intensity of the solar 
#spectrum in the region of the Fe I absorption line @6173A on the surface of the sun.
#The continuum data is available in the DRMS series  hmi.Ic_45s and hmi.Ic_720s
#Continuum data with Limb Darkening removed can be found in the DRMS series hmi.Ic_noLimbDark_720s
###############################################################################

# si = client.info('hmi.Ic_720s')                         #data series of interest
# qstr = 'hmi.Ic_720s[2022.02.02_TAI-2022.02.03_TAI@24h]'        #start/end/cadence

# si = client.info('hmi.Ic_720s')                         #data series of interest
# qstr = 'hmi.Ic_720s[2022.02.02_TAI-2022.02.03_TAI@24h]' 

# print(f'Data export query:\n {qstr}\n')

# jpg_args = {                              #construct dictionary for jpeg export
#     'ct': 'grey.sao',
#     'min': 10000,
#     'max': 50000,
#     'scaling': 'MINMAXGIVEN',
#     'size': 1,
#     }

# process = {'im_patch':{     #construct dictionary specyfying the cutout request
#     't_ref': '2022-02-05T13:22:08',
#     't': 0,
#     'r': 0,
#     'c': 0,
#     'locunits': 'arcsec',
#     'boxunits': 'arcsec',
#     'x': 0,
#     'y': -166,
#     'width': 50,
#     'height': 50,
#     }}


# print('Submitting export request...')
# result = client.export(     # Submit request using 'process' to define a cutout
#     qstr,
#     method = 'url',
#     protocol='fits',
#     # protocol = 'jpg',
#     # protocol_args = jpg_args,
#     email = 'leiftinwell@rocketmail.com',
#     process = process,
#     )

# keyword_result = client.query(                 #query datetime of recorded data
#     qstr,
#     key=['T_REC'],
#     )

# datamean_result = client.query(      #query DATAMEAN for each image in sequence
#     qstr,
#     key=['DATAMEAN'],
#     )



###############################################################################
#compute umbral/penumbral pixels from DATAMEAN values such that Umbral pixel is less or equal to 
#0.6 of the average DATANEAN
#and Penumbral pixels b/w 0.6 and 1.05 average DATAMEAN
###############################################################################

# TO BE DONE, use computed values as thresholds for ellipse fitting routine
# def av_pix():
#     pix_list = datamean_result
#     pix_data = pd.DataFrame(pix_list).to_numpy()
#     av_pix = math.trunc(np.mean(pix_data))
#     return av_pix
# # print(av_pix())
        
# print(f' -> {int(len(keyword_result))} lines retrieved.')

# result.index = drms.to_datetime(result.pop('T_REC')) 
# time_data = result.index         #convert T_REC to datetime type and index list
# print(time_data)
# print(f'\nRequest URL: {result.request_url}')                #print request URL
# print(f'{int(len(result.urls))} file(s) available for download.\n')

# result.wait()
# downloaded_file = result.download(out_dir)     #download file from request URLs

# print('Download finished.')
# print(f'\nDownload directory:\n "{os.path.abspath(out_dir)}"\n')





###############################################################################
#Need to rotate the image such that the solar North is pointed up. The roll angle
#of the instrument is reported in the FITS header keyword 'CROTA2' stating that 
#the nominal CROTA2 for HMI is =179.93"
#Plotting of data requires .FITS files. Use for report, otherwise useless to the script 
###############################################################################

# def image_manip(filename):
#     for filename in os.listdir(directory):
#         file = os.path.join(directory, filename)
#         if filename.endswith('.fits'):
#             print(file)
#             hmi_map = sunpy.map.Map(file)
#             hmi_map.plot(autoalign=True)
#             plt.show()

###############################################################################
#In order for the rotation of the sunspot to be measured, umbral and penumbral 
#boundaries are determined by using a combination of Gaussian blurs 
#and intensity thresholds. Thresholds are then used to contruct umbral/penumbral 
#boundaries, which are themselves used to compute the center of mass of the sunspot.
#An ellipse is fitted to the contours, using the center of mass as its center.
#The angle of rotation from vertical is computed using the coordinates of the 
#semi-major axis. Iterated for each image of the sequence and plotted as a rotation 
#profile over time.
###############################################################################  

# umbra_angle_data = []                                   #initializing variables
# penumbra_angle_data = []                                                  #idem

# def penumbra_ellipse_fitting():
#     for filename in os.listdir(directory):
#         file = os.path.join(directory, filename)
#         im = cv.imread(file)                      #collect files from directory
#         imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)       #convert to grayscale
#         blur = cv.GaussianBlur(imgray, (25,25), 0)
#         #threshold to eliminate excess data, focus on penumbra area of sunspot
#         threshold = cv.threshold(blur, 185, 255, cv.THRESH_BINARY_INV)[1] 
#         #clean the silhouette to make ellipse fitting more efficient
#         morph1 = cv.morphologyEx(threshold, 
#                                   cv.MORPH_CLOSE,
#                                   cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
#                                   ) 
#         morph2 = cv.morphologyEx(morph1, 
#                                   cv.MORPH_OPEN, 
#                                   cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
#                                   )
#           #find contours of threshold silhouette
#         contours = cv.findContours(morph2, 
#                                     cv.RETR_EXTERNAL, 
#                                     cv.CHAIN_APPROX_NONE
#                                     )
#         contours = contours[0] if len(contours) == 2 else contours[1]
#         big_contour = max(contours, key=cv.contourArea)
#         ellipse = cv.fitEllipse(big_contour)       #fit ellipse around contours
#         (xc,yc),(d1,d2),angle = ellipse
#         result = im.copy()
#         cv.ellipse(result, ellipse, (0, 255, 0), 1)
#         xc, yc = ellipse[0]
#         cv.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot
#         penumbra_angle_data.append(angle)        #send angle data to angle_data 

# def umbra_ellipse_fitting():
#     for filename in os.listdir(directory):
#         file = os.path.join(directory, filename)
#         im = cv.imread(file)                      #collect files from directory
#         imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)       #convert to grayscale
#         blur = cv.GaussianBlur(imgray, (25,25), 0)
#         #threshold to eliminate excess data, focus on umbra area of sunspot
#         threshold = cv.threshold(blur, 75, 255, cv.THRESH_BINARY_INV)[1] 
#         morph1 = cv.morphologyEx(threshold, 
#                                   cv.MORPH_CLOSE,
#                                   cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
#                                   )
#         morph2 = cv.morphologyEx(morph1, 
#                                   cv.MORPH_OPEN, 
#                                   cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
#                                   )
#         contours = cv.findContours(morph2, 
#                                     cv.RETR_EXTERNAL, 
#                                     cv.CHAIN_APPROX_NONE
#                                     )
#         contours = contours[0] if len(contours) == 2 else contours[1]
#         big_contour = max(contours, key=cv.contourArea)
#         ellipse = cv.fitEllipse(big_contour)
#         (xc,yc),(d1,d2),angle = ellipse
#         result = im.copy()
#         cv.ellipse(result, ellipse, (0, 255, 0), 1)
#         xc, yc = ellipse[0]
#         cv.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)
#         umbra_angle_data.append(angle)

# umbra_ellipse_fitting()
# penumbra_ellipse_fitting()
# fig, ax = plt.subplots(constrained_layout = True)  
# ax.set(title="The Devil's Asshole Rotation Profile")
# ax.plot(time_data, umbra_angle_data, color='lime', label='Umbral Rotation')
# ax.plot(time_data, penumbra_angle_data, color='red', label='Penumbral Rotation')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
# ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
# ax.set_xlabel('Time')
# ax.set_ylabel('Angle (θ)')
# ax.grid(False)
# ax.legend(loc='upper left');
# plt.show()




