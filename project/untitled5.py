# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 01:35:51 2022

@author: Leif Tinwell
"""
from sunpy.map import Map
from sunpy.net.helioviewer import HelioviewerClient
import os 
import matplotlib.pyplot as plt 
import astropy.time
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, SqrtStretch
from sunpy.net import Fido 
from sunpy.net import jsoc, fido_factory, attrs as a
import drms
from astropy.io import fits 
from astropy.wcs import WCS 
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
hv = HelioviewerClient()
datasources = hv.get_data_sources()
# print(datasources)

    
# '2017-12-24T20:34:07'
# filepath = hv.download_jp2( '2017/10/27 23:33:27' , source_id=(18))
filepath = hv.download_png('2017/10/27 23:33:27' , 2, '[18, 1, 100]', x0=0, y0=0, width=4096, height=4096 )

# hmi = Map(filepath)
# hmi_cutout = hmi.submap(
#     bottom_left=SkyCoord(-34*u.arcsec, -500*u.arcsec, frame=hmi.coordinate_frame),
#     top_right=SkyCoord(19*u.arcsec, -458*u.arcsec, frame=hmi.coordinate_frame)
#     )
# hmi.peek()
# plt.figure()
# hmi_cutout.plot()
# plt.show()
