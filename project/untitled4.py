# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:19:50 2022

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
from astropy.coordinates import SkyCoord
import sunpy.map 
from sunpy.net import Fido
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import ImageNormalize, SqrtStretch
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

###############################################################################
client = drms.Client(email = 'leiftinwell@rocketmail.com', verbose=True)
out_dir = 'downloads'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
fits_dir = os.fsdecode('D:/Desktop/Uni/project/downloads/')
# jpg_args = {
#     'ct': 'HMI_mag.lut',
#     'min': 1,
#     'max': 800,
#     'scaling': 'log',
#     'size': 2,
# }

si = client.info('hmi.Ic_720s')
qstr = 'hmi.Ic_720s[2021.12.24_TAI-2021.12.26_TAI@24h]'
print(f'Data export query:\n {qstr}\n')

#construct dictionary specyfying the cutout request
process = {'im_patch':{
    't_ref': '2021.12.25T22:33:22',
    't': 0,
    'r': 0,
    'c': 0,
    'locunits': 'arcsec',
    'boxunits': 'arcsec',
    'x': -418,
    'y': -254,
    'width': 300,
    'height': 300,
    }}

#submit request using 'fits' or 'jpg' protocol
print('Submitting export request...')
result = client.export(
    qstr,
    method='url',
    protocol='fits',
    email = 'leiftinwell@rocketmail.com',
    process = process,
    )

print(result)

#print request url
print(f'\nRequest URL: {result.request_url}')
print(f'{int(len(result.urls))} file(s) available for download.\n')

#download requested files =
# result.wait()
# downloaded_file = result.download(out_dir)
print('Download finished.')
print(f'\nDownload directory:\n "{os.path.abspath(out_dir)}"\n')


###############################################################################
#Need to rotate the image such that the solar North is pointed up. The roll angle
#of the instrument is reported in the FITS header keyword 'CROTA2' stating that 
#the nominal CROTA2 for HMI is =179.93"


directory = 'downloads'
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    if filename.endswith('.fits'):
        print(file)
        hmi_map = sunpy.map.Map(file)
        hmi_map.plot()
        hmi_mapRot = hmi_map.rotate(order=3)
        hmi_mapRot.plot()
        plt.show()

# for file in os.listdir(fits_dir):
#     filename = os.fsdecode(file)
#     if filename.endswith(".fits"):
#         hmi_map = sunpy.map.Map(fits_dir)
#         hmi_map.plot()
#         plt.show()
#         continue


# hmi_map = sunpy.map.Map(os.path.abspath(out_dir))
# hmi_map.plot()
# plt.show()
# plt.show()
# hmi_map = hmi_map.rotate(order=3)
# hmi_map.save('*.fits')








