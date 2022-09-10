# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:45:27 2022

@author: Leif Tinwell
"""

import os
from os import listdir
import drms
import matplotlib.pyplot as plt 
import matplotlib.animation as anim
import sys
import cv2
import numpy as np
import matplotlib.cm as cm
import sunpy.map
from sunpy.net import Fido
from astropy.io import fits
from astropy.wcs import WCS
from sunpy.map import Map
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import ImageNormalize, SqrtStretch
import matplotlib.animation as animation
# file = r"D:\Desktop\Uni\project\downloads\.fits"
# filePath = r"D:\Desktop\Uni\project\plots"
# hmi_map = Map(file)
# hmi_map.plot()
# plt.show()
# hmi_map.save('map_{index}.fits')

file = 'downloads'

# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     if filename.endswith('.fits'):
#         print(f)
#         hmi_map = Map(f)
#         hmi_map.plot()
#         hmi_mapRot = hmi_map.rotate(order=3)
#         hmi_mapRot.plot()
#         plt.show()
        
# sequence = Map(directory, sequence=True)
# ani = sequence.peek()   
# plt.show()

# def open_image(file):
#     hdu = fits.open(file)
#     hdu = montage.reproject_hdu(hdu[0], north_alligned=True)
#     image = hdu.data
#     nans = np.isnan(image)
#     image[nans] = 0
#     header = hdu.header
#     wcs = WCS(header)
#     return image, header, wcs

# def plot_image(file):
#     image, header, wcs = open_image(file) #load image
#     print(wcs)
    


hdulist = fits.open(file)
hdu = hdulist['DATAMEAN']
hdu.data.shape
hdu.header



























