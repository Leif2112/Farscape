# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:30:56 2022

@author: Leif Tinwell
"""
import os
from os import listdir
import drms
import cv2
import gc  
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
plt.ioff()
from matplotlib.ticker import MultipleLocator
import sunpy.map
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

    # -------------------------------------------------------------------------
 
# initializing DRMS client to given email address
client = drms.Client(email = 'leiftinwell@rocketmail.com', verbose=True)

fits_dir = 'D:/Desktop/Uni/project/downloads/'
corrected = 'D:/Desktop/Uni/project/corrected/' 
out_dir = 'downloads'

    # -------------------------------------------------------------------------


"""
The HMI continuum data refers to the map of the continuum intensity of the solar 
spectrum in the region of the Fe I absorption line @6173A on the surface of the sun.
The continuum data is available in the DRMS series  hmi.Ic_45s and hmi.Ic_720s
Continuum data with Limb Darkening removed can be found in the DRMS series hmi.Ic_noLimbDark_720s
"""

si = client.info('hmi.Ic_45s')                         #data series of interest
qstr = 'hmi.Ic_45s[2022.02.06_TAI-2022.02.07_TAI@24h]'        #start/end/cadence

print(f'Data export query:\n {qstr}\n')

process = {'im_patch':{     #construct dictionary specyfying the cutout request
    't_ref': '2011-01-21T07:16:35',
    't': 0,
    'r': 0,
    'c': 0,
    'locunits': 'arcsec',
    'boxunits': 'arcsec',
    'x': -20,
    'y': 469,
    'width': 50,
    'height': 50,
    }}


print('Submitting export request...')
result = client.export(     # Submit request using 'process' to define a cutout
    qstr,
    method = 'url',
    protocol='fits',
    email = 'leiftinwell@rocketmail.com',
    # process = process,
    )

print(result)
print(f'\nRequest URL: {result.request_url}')                #print request URL
print(f'{int(len(result.urls))} file(s) available for download.\n')
result.wait()
downloaded_file = result.download(out_dir)     #download file from request URLs
print('Download finished.')
print(f'\nDownload directory:\n "{os.path.abspath(out_dir)}"\n')


def image_manip():
    for filename in os.listdir(fits_dir):
        file = os.path.join(fits_dir, filename)
        if filename.endswith('.fits'):
            print(file)
            hmi_map = sunpy.map.Map(file)
            # hmi_map.plot(autoalign=True)
            hmi_map.peek()


"""
Once files for a given sunspot are downloaded, they need to be processed to eliminate 
the effects of Limb Darkening.

    Based on:
      https://hesperia.gsfc.nasa.gov/ssw/gen/idl/solar/darklimb_correct.pro

    Coefficients taken from 
      Cox, A. N.: Allen's Astrophysical Quantities, Springer, 2000 taken from IDL
"""

def writefits(image):
    os.system("rm -r limbcorrect.fits")
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdul.writeto('limbcorrected.fits')
    

def figure(image, title='image'):
    fig = plt.figure(figsize=(8,8), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image, cmap = 'Greys_r',origin='lower')
    plt.savefig(title + ".png", dpi='figure')
    plt.close(fig)
    

def darklimb(array):
    
    """
    Darklimb function:
        files need to be .FITS in order to take advantage of Header structure
        
    Output:
        DOS array: First image is the corrected array, the second is the original.
    """
    
    # -------------------------------------------------------------------------
    
    def darklimb_u(ll):
        pll = np.array([1.0,ll,ll**2,ll**3,ll**4,ll**5])
        au = -8.9829751
        bu = 0.0069093916
        cu = -1.8144591e-6
        du = 2.2540875e-10
        eu = -1.3389747e-14
        fu = 3.0453572e-19
        aa=np.array([au,bu,cu,du,eu,fu])
        ul = sum(aa*pll)
        return ul
    
    def darklimb_v(ll):
        pll = np.array([1.0,ll,ll**2,ll**3,ll**4,ll**5])
        av = 9.2891180
        bv = -0.0062212632
        cv = 1.5788029e-6
        dv = -1.9359644e-10
        ev = 1.1444469e-14
        fv = -2.599494e-19
        aa=np.array([av,bv,cv,dv,ev,fv])
        vl = sum(aa*pll)
        return vl
    
    # -------------------------------------------------------------------------

    #Read data files
    data = fits.getdata(name, 1)
    head = fits.getheader(name, 1)
    
    #Parameters needed for the function
    wavelength = head['WAVELNTH'] # Wavelength 
    xcen = head['CRPIX1'] # X center
    ycen = head['CRPIX2'] # Y center
    radius = head['RSUN_OBS']/head['CDELT1'] # Pixels result
    size = head['NAXIS1'] # X array size
    
    ll = 1.0*wavelength
    
    array = np.array(data) # convert data into np array 
    NaNs = np.isnan(array) # look for NaNs
    array[NaNs] = 0.0      # Make all NaNs = 0
    
    #Apply correction
    ul = darklimb_u(ll)
    vl = darklimb_v(ll)
    
    xarr = np.arange(0,size,1.0) # Make X array
    yarr = np.arange(0,size,1.0) # Make Y array
    xx, yy = np.meshgrid(xarr, yarr) # Make XY array
    # z: make array so that the zero center is the XY center
    # Make circle
    z = np.sqrt((xx-xcen)**2 + (yy-ycen)**2) 
    # grid: Normalize the circle so that inside radius is the unity
    grid = z/radius 
    out = np.where(grid>1.0) # Look for the values greater than unity
    grid[out] = 0.0 # Make zero all those values (Domain of arcsin)
    
    limbfilt =  1.0-ul-vl+ul*np.cos(np.arcsin(grid))+vl*np.cos(np.arcsin(grid))**2
    
    # Final image
    imgout = np.array(array/limbfilt)
    
    return imgout, array

    # -------------------------------------------------------------------------

for filename in os.listdir(fits_dir):
    name = os.path.join(fits_dir, filename)
    print(name)
    new_name = 'corrected/' + str(filename)
    corrected, original = darklimb(name)
    figure(corrected, title = new_name)

"""
In order for the rotation of the sunspot to be measured, umbral and penumbral 
boundaries are determined by using a combination of Gaussian blurs 
and intensity thresholds. Thresholds are then used to contruct umbral/penumbral 
boundaries, which are themselves used to compute the center of mass of the sunspot.
An ellipse is fitted to the contours.
The angle of rotation is computed using the coordinates of the 
semi-major axis. Iterated for each image of the sequence and plotted as a rotation 
profile over time.
""" 

                                 
penumbra_angle_data = []
umbra_angle_data = []                               #initializing variables                     
umbra_angle_dataUB = []
umbra_angle_dataMB = []
umbra_angle_dataLB = []
time_data = []
cpad = []
cuad= []
cpad1 = []
cuad1= []
cpad2 = []
cuad2= []
cpad3 = []
cuad3= []
Parea = []
Uarea = []
area_val1 = []
area_val2 = []
area_val3 = []


# def grab_time_data():
#     for filename in os.listdir('D:/Desktop/Uni/project/AR2939/.FITS Data/'):
#         name = os.path.join('D:/Desktop/Uni/project/AR2939/.FITS Data/', filename)
#         hdulist = fits.open(name)
#         hdu = hdulist[1]
#         t_rec = drms.to_datetime(hdu.header['T_REC'])
#         time_data.append(t_rec)

# def thresh_val():
#     datamean_result = client.query(      #query DATAMEAN for each image in sequence
#     qstr,key=['DATAMEAN'])
#     dtm = pd.DataFrame(datamean_result)
#     print(dtm)
#     datamean = dtm.mean()
#     print(f'datamean mean: {datamean}')
#     IUmbral_UB = 0.55*datamean
#     IUmbral_LB = 0.125*datamean
#     IPenumbral_UB = 1.30*datamean
#     IPenumbral_LB = 0.6*datamean
#     val = {'umbra upper bound': IUmbral_UB, 
#             'umbra lower bound': IUmbral_LB,
#             'penumbral upper bound': IPenumbral_UB,
#             'penumbral lower bound': IPenumbral_LB
#             }
#     return val    
# I = thresh_val()
# Umbral_lowerbound = I['umbra lower bound']/255
# Umbral_upperbound = I['umbra upper bound']/255
# Umbral_MB = Umbral_lowerbound*2
# Penumbral_lowerbound = I['penumbral lower bound']/255
# Penumbral_upperbound = I['penumbral upper bound']/255
# print(f'\n Umbral lower bound: {Umbral_lowerbound} \n', 
#       f'\n Umbral upper bound: {Umbral_upperbound} \n',
#       f'\n Penumbral lower bound: {Penumbral_lowerbound} \n',
#       f'\n Penumbral upper bound: {Penumbral_upperbound} \n'
#       )

# def penumbra_ellipse_fitting():
#     for filename in os.listdir('D:/Desktop/Uni/project/AR2939/Image Data/'):
#         file = os.path.join('D:/Desktop/Uni/project/AR2939/Image Data/', filename)
#         im = cv2.imread(file)                      #collect files from directory
#         imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       #convert to grayscale
#         blur = cv2.GaussianBlur(imgray, (45,45),55)
#         #threshold to eliminate excess data, focus on penumbra area of sunspot
#         threshold = cv2.threshold(blur, 215, 255, cv2.THRESH_BINARY_INV)[1]
#           #find contours of threshold silhouette
#         contours = cv2.findContours(threshold, 
#                                     cv2.RETR_EXTERNAL, 
#                                     cv2.CHAIN_APPROX_NONE
#                                     )
#         contours = contours[0] if len(contours) == 2 else contours[1]
#         big_contour = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(big_contour)
#         Parea.append(area)
#         ellipse = cv2.fitEllipse(big_contour)       #fit ellipse around contours
#         (xc,yc),(d1,d2),anglep = ellipse
#         result = im.copy()
#         cv2.ellipse(result, ellipse, (0, 255, 0), 1)
#         xc, yc = ellipse[0]
#         cv2.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot
#         penumbra_angle_data.append(anglep)        #send angle data to angle_data 
        
# def umbra_ellipse_fitting():
#     for filename in os.listdir('D:/Desktop/Uni/project/AR2939/Image Data/'):
#         file = os.path.join('D:/Desktop/Uni/project/AR2939/Image Data/', filename)
#         im = cv2.imread(file)                      #collect files from directory
#         imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       #convert to grayscale
#         blur = cv2.GaussianBlur(imgray, (25,25), 25)
#         #threshold to eliminate excess data, focus on penumbra area of sunspot
#         threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)[1] 
#         #find contours of threshold silhouette
#         contours = cv2.findContours(threshold, 
#                                     cv2.RETR_TREE, 
#                                     cv2.CHAIN_APPROX_NONE
#                                     )
#         contours = contours[0] if len(contours) == 2 else contours[1]
#         big_contour = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(big_contour)
#         Uarea.append(area)
#         ellipse = cv2.fitEllipse(big_contour)       #fit ellipse around contours
#         (xc,yc),(d1,d2),angleu = ellipse
#         result = im.copy()
#         cv2.ellipse(result, ellipse, (0, 255, 0), 1)
#         xc, yc = ellipse[0]
#         cv2.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot
#         umbra_angle_data.append(angleu)        #send angle data to angle_data

        

# #     # -------------------------------------------------------------------------
    
# grab_time_data()
# time_since_start = []
# date_format = '%Y-%m-%d %H:%M:%S'
# for i in time_data:
#     t0 = datetime.strptime(str(time_data[0]), date_format)
#     ti = datetime.strptime(str(i), date_format)
#     td = ti - t0
#     tdm = td.total_seconds() / 3600
#     time_since_start.append(tdm)
    
#     # -------------------------------------------------------------------------
    
# penumbra_ellipse_fitting()
# #cumulative rotation starting from 0
# for i in penumbra_angle_data:
#     a0 = float(penumbra_angle_data[0])
#     ai = float(i)
#     ca = ai - a0
#     cpad.append(ca)  
# cpad = pd.Series(cpad)
# prp = cpad.rolling(60)
# prpm = prp.mean()

# umbra_ellipse_fitting()
# #cumulative rotation starting from 0
# for i in umbra_angle_data:
#     a0 = float(umbra_angle_data[0])
#     ai = float(i)
#     ca = ai - a0
#     cuad.append(ca)  
# cuad = pd.Series(cuad)
# urp = cuad.rolling(20)
# urpm = urp.mean()

#     # -------------------------------------------------------------------------
# fig, ax1 = plt.subplots(constrained_layout=True)
# ax1.set(title='AR 2939 area plot 02/02/2022-08/02/2022')
# ax1.plot(time_since_start, Uarea, color='#39817f', label='Umbral area')
# ax1.plot(time_since_start, Parea, color='#a9c12a', label='Penumbral area')
# ax1.set_xlabel('Time (hours)')
# ax1.set_ylabel('Area (pixels)')
# ax1.minorticks_on()
# ax1.xaxis.set_major_locator(MultipleLocator(10))
# ax1.xaxis.set_minor_locator(MultipleLocator(2))
# ax1.xaxis.grid(False)
# ax1.yaxis.grid(False)
# ax1.tick_params(which='major', width=2)
# ax1.tick_params(which='major', length=7)
# ax1.tick_params(which='minor', length=3)
# ax1.legend(loc='upper right')
# fig.savefig('AR2939UPA120215.png', dpi='figure')

#     # -------------------------------------------------------------------------

# fig, ax2 = plt.subplots(constrained_layout=True)
# ax2.set(title='AR 2939 Penumbral Rotation 02/02/2022-08/02/2022')
# ax2.plot(time_since_start, cpad, color='gainsboro', label='Raw Data', linewidth=3)
# ax2.plot(time_since_start, prpm, color='#a9c12a', label='Rolling Average')
# plt.axhline(y=0, color='lightgray', linestyle='--')
# ax2.set_xlabel('Time (hours)')
# ax2.set_ylabel('Angle ( °)')
# ax2.minorticks_on()
# ax2.xaxis.set_major_locator(MultipleLocator(10))
# ax2.xaxis.set_minor_locator(MultipleLocator(2))
# ax2.yaxis.set_major_locator(MultipleLocator(10))
# ax2.yaxis.set_minor_locator(MultipleLocator(2))
# ax2.xaxis.grid(False)
# ax2.yaxis.grid(False)
# ax2.tick_params(which='major', width=2)
# ax2.tick_params(which='major', length=7)
# ax2.tick_params(which='minor', length=3)
# ax2.legend(loc='lower left')
# fig.savefig('AR2939PR.png', dpi='figure')
    
#     # -------------------------------------------------------------------------

# fig, ax1 = plt.subplots(constrained_layout=True)
# ax1.set(title='AR 2939 Umbral Rotation 02/02/2022-08/02/2022')
# ax1.plot(time_since_start, cuad, color='silver', label='Raw data', linewidth=3)
# ax1.plot(time_since_start, urpm, color='#39817f', label='Rolling Average')
# plt.axhline(y=0, color='lightgray', linestyle='--')
# plt.axvline(x=90, color='lightgray', linestyle='dashdot')
# plt.axvline(x=118, color='lightgray', linestyle='dashdot')
# ax1.set_xlabel('Time (hours)')
# ax1.set_ylabel('Angle ( °)')
# ax1.minorticks_on()
# ax1.xaxis.set_major_locator(MultipleLocator(10))
# ax1.xaxis.set_minor_locator(MultipleLocator(2))
# ax1.yaxis.set_major_locator(MultipleLocator(10))
# ax1.yaxis.set_minor_locator(MultipleLocator(2))
# ax1.xaxis.grid(False)
# ax1.yaxis.grid(False)
# ax1.tick_params(which='major', width=2)
# ax1.tick_params(which='major', length=7)
# ax1.tick_params(which='minor', length=3)
# ax1.legend(loc='lower left')
# fig.savefig('AR2939UR.png', dpi='figure')
