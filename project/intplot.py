# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:35:11 2022

@author: Leif Tinwell
"""

import os
from os import listdir
import drms
import math
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib
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
from matplotlib.patches import ConnectionPatch
plt.style.use(astropy_mpl_style)


t0 = astropy.time.Time('2022-02-05T13:22:08', scale='utc', format='isot')

#query full frame HMI image from fido for quick visualization of targeted sunspot
q = Fido.search(   
    a.Instrument.hmi,
    a.Physobs.intensity,
    a.Time(t0, t0 + 10*u.s)
    )
qf = Fido.fetch(q)
m = sunpy.map.Map(qf).rotate(order=3)

left_corner = SkyCoord(Tx=-32*u.arcsec, Ty=-142*u.arcsec, frame=m.coordinate_frame)
right_corner = SkyCoord(Tx=30*u.arcsec, Ty=-187*u.arcsec, frame=m.coordinate_frame)

hpc_coords = sunpy.map.all_coordinates_from_map(m)
mask = ~sunpy.map.coordinate_is_on_solar_disk(hpc_coords)
m_big = sunpy.map.Map(m.data, m.meta, mask=mask)

fig = plt.figure(figsize=(7.2, 4.8))
norm = matplotlib.colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)
ax1 = fig.add_subplot(121, projection=m_big)
m.plot(axes=ax1, cmap='Greys_r', annotate=False,)
m.draw_grid(axes=ax1, color='black', alpha=0.25, lw=0.5)

for coord in ax1.coords:
    coord.frame.set_linewidth(0)
    coord.set_ticks_visible(False)
    coord.set_ticklabel_visible(False)

m_big.draw_quadrangle(left_corner, top_right=right_corner, edgecolor='black', lw=1)
m_small = m.submap(left_corner, top_right=right_corner)
ax2 = fig.add_subplot(122, projection=m_small)
im = m_small.plot(axes=ax2, cmap='Greys_r', annotate=False,)
ax2.grid(alpha=0)
lon, lat = ax2.coords[0], ax2.coords[1]
lon.frame.set_linewidth(1)
lat.frame.set_linewidth(1)
lon.set_axislabel('Helioprojective Longitude',)
lon.set_ticks_position('b')
lat.set_axislabel('Helioprojective Latitude',)
lat.set_axislabel_position('r')
lat.set_ticks_position('r')
lat.set_ticklabel_position('r')

xpix, ypix = m_big.world_to_pixel(right_corner)
con1 = ConnectionPatch(
    (0, 1), (xpix.value, ypix.value), 'axes fraction', 'data', axesA=ax2, axesB=ax1,
    arrowstyle='-', color='black', lw=1
)
xpix, ypix = m_big.world_to_pixel(
    SkyCoord(right_corner.Tx, left_corner.Ty, frame=m_big.coordinate_frame))
con2 = ConnectionPatch(
    (0, 0), (xpix.value, ypix.value), 'axes fraction', 'data', axesA=ax2, axesB=ax1,
    arrowstyle='-', color='black', lw=1
)
ax2.add_artist(con1)
ax2.add_artist(con2)

pos = ax2.get_position().get_points()
cax = fig.add_axes([
    pos[0, 0], pos[1, 1]+0.01, pos[1, 0]-pos[0, 0], 0.025
])
