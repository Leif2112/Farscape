# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:30:56 2022

@author: Leif Tinwell
"""
import os
from os import listdir
import drms
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator
import sunpy.map
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

    # -------------------------------------------------------------------------
 
# # initializing DRMS client to given email address
# client = drms.Client(email = 'leiftinwell@rocketmail.com', verbose=True)

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

# si = client.info('hmi.Ic_45s')                         #data series of interest
# qstr = 'hmi.Ic_45s[2022.02.02_TAI-2022.02.08_TAI@3m]'        #start/end/cadence

# print(f'Data export query:\n {qstr}\n')

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
#     email = 'leiftinwell@rocketmail.com',
#     process = process,
#     )

# print(result)
# print(f'\nRequest URL: {result.request_url}')                #print request URL
# print(f'{int(len(result.urls))} file(s) available for download.\n')
# result.wait()
# downloaded_file = result.download(out_dir)     #download file from request URLs
# print('Download finished.')
# print(f'\nDownload directory:\n "{os.path.abspath(out_dir)}"\n')


"""
Need to rotate the image such that the solar North is pointed up. The roll angle
of the instrument is reported in the FITS header keyword 'CROTA2' stating that 
the nominal CROTA2 for HMI is =179.93"

Plotting of data requires .FITS files. Use for report, otherwise useless to the script 
"""

# def image_manip(filename):
#     for filename in os.listdir(directory):
#         file = os.path.join(directory, filename)
#         if filename.endswith('.fits'):
#             print(file)
#             hmi_map = sunpy.map.Map(file)
#             hmi_map.plot(autoalign=True)
#             plt.show()


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
    

def figure(image, title='image', save=False):
    fig = plt.figure(figsize=(8,8), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plt.imshow(image,cmap = 'Greys_r',origin='lower')
    if save == True:
        fig.savefig(title + ".png")
    # else:
    #     pass
    # plt.show()
    

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
    figure(corrected, title = new_name, save=True)

"""
In order for the rotation of the sunspot to be measured, umbral and penumbral 
boundaries are determined by using a combination of Gaussian blurs 
and intensity thresholds. Thresholds are then used to contruct umbral/penumbral 
boundaries, which are themselves used to compute the center of mass of the sunspot.
An ellipse is fitted to the contours, using the center of mass as its center.
The angle of rotation from vertical is computed using the coordinates of the 
semi-major axis. Iterated for each image of the sequence and plotted as a rotation 
profile over time.

""" 

                                 
penumbra_angle_data = []
umbra_angle_data = []                               #initializing variables                     
time_data = []
cpad = []
cuad= []
def grab_time_data():
    for filename in os.listdir(fits_dir):
        name = os.path.join(fits_dir, filename)
        hdulist = fits.open(name)
        hdu = hdulist[1]
        t_rec = drms.to_datetime(hdu.header['T_REC'])
        time_data.append(t_rec)


def penumbra_ellipse_fitting():
    for filename in os.listdir('D:/Desktop/Uni/project/corrected/'):
        file = os.path.join('D:/Desktop/Uni/project/corrected/', filename)
        im = cv2.imread(file)                      #collect files from directory
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       #convert to grayscale
        blur = cv2.GaussianBlur(imgray, (45,45),55)
        #threshold to eliminate excess data, focus on penumbra area of sunspot
        threshold = cv2.threshold(blur, 215, 255, cv2.THRESH_BINARY_INV)[1]
          #find contours of threshold silhouette
        contours = cv2.findContours(threshold, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_NONE
                                    )
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(big_contour)       #fit ellipse around contours
        (xc,yc),(d1,d2),anglep = ellipse
        result = im.copy()
        cv2.ellipse(result, ellipse, (0, 255, 0), 1)
        xc, yc = ellipse[0]
        cv2.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot
        penumbra_angle_data.append(anglep)        #send angle data to angle_data 
        
def umbra_ellipse_fitting():
    for filename in os.listdir('D:/Desktop/Uni/project/corrected/'):
        file = os.path.join('D:/Desktop/Uni/project/corrected/', filename)
        im = cv2.imread(file)                      #collect files from directory
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       #convert to grayscale
        blur = cv2.GaussianBlur(imgray, (25,25), 0)
        #threshold to eliminate excess data, focus on penumbra area of sunspot
        threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
        #find contours of threshold silhouette
        contours = cv2.findContours(threshold, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_NONE
                                    )
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(big_contour)       #fit ellipse around contours
        (xc,yc),(d1,d2),angleu = ellipse
        result = im.copy()
        cv2.ellipse(result, ellipse, (0, 255, 0), 1)
        xc, yc = ellipse[0]
        cv2.circle(result, (int(xc),int(yc)), 1, (255, 255, 255), -1)#center dot
        umbra_angle_data.append(angleu)        #send angle data to angle_data

    # -------------------------------------------------------------------------
    
grab_time_data()
time_since_start = []
date_format = '%Y-%m-%d %H:%M:%S'
for i in time_data:
    t0 = datetime.strptime(str(time_data[0]), date_format)
    ti = datetime.strptime(str(i), date_format)
    td = ti - t0
    tdm = td.total_seconds() / 3600
    time_since_start.append(tdm)
    
    # -------------------------------------------------------------------------
    
penumbra_ellipse_fitting()
#cumulative rotation starting from 0
for i in penumbra_angle_data:
    a0 = float(penumbra_angle_data[0])
    ai = float(i)
    ca = ai - a0
    cpad.append(ca)  
cpad = pd.Series(cpad)
prp = cpad.rolling(20)
prpm = prp.mean()
cump0 = cpad.expanding().mean()

umbra_ellipse_fitting()
#cumulative rotation starting from 0
for i in umbra_angle_data:
    a0 = float(umbra_angle_data[0])
    ai = float(i)
    ca = ai - a0
    cuad.append(ca)  
cuad = pd.Series(cuad)
urp = cuad.rolling(20)
urpm = urp.mean()
cumu0 = cuad.expanding().mean()


    # -------------------------------------------------------------------------

fig, ax2 = plt.subplots(constrained_layout=True)
ax2.set(title='AR 2939 Penumbral Rotation')
ax2.plot(time_since_start, cpad, color='silver', label='Raw Data', linewidth=3)
ax2.plot(time_since_start, prpm, color='slateblue', label='Rolling Average')
ax2.plot(time_since_start, cump0, color='crimson', label='Cumulative')
plt.axhline(y=0, color='lightgray', linestyle='--')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Angle ( °)')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(10))
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.yaxis.set_major_locator(MultipleLocator(10))
ax2.yaxis.set_minor_locator(MultipleLocator(2))
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)
ax2.tick_params(which='major', width=2)
ax2.tick_params(which='major', length=7)
ax2.tick_params(which='minor', length=3)
ax2.legend(loc='lower left')
plt.show()
    
    # -------------------------------------------------------------------------
    
fig, ax1 = plt.subplots(constrained_layout=True)
ax1.set(title='AR 2939 Umbral Rotation')
ax1.plot(time_since_start, cuad, color='silver', label='Raw Data', linewidth=3)
ax1.plot(time_since_start, urpm, color='slateblue', label='Rolling Average')
ax1.plot(time_since_start, cumu0, color='crimson', label='Cumulative')
plt.axhline(y=0, color='lightgray', linestyle='--')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Angle ( °)')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(10))
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.yaxis.set_minor_locator(MultipleLocator(2))
ax1.xaxis.grid(False)
ax1.yaxis.grid(False)
ax1.tick_params(which='major', width=2)
ax1.tick_params(which='major', length=7)
ax1.tick_params(which='minor', length=3)
ax1.legend(loc='lower left')
plt.show()

    # -------------------------------------------------------------------------
    
fig, ax3 = plt.subplots(constrained_layout=True)
ax3.set(title='AR 2939 Umbral And Penumbral Rotation')
ax3.plot(time_since_start, cumu0, color='dimgrey', label='Cumulative Umbral Rotation')
ax3.plot(time_since_start, cump0, color='darkgray', label='Cumulative Penumbral Rotation')
plt.axhline(y=0, color='lightgray', linestyle='--')
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Angle ( °)')
ax3.minorticks_on()
ax3.xaxis.set_major_locator(MultipleLocator(10))
ax3.xaxis.set_minor_locator(MultipleLocator(2))
ax3.yaxis.set_major_locator(MultipleLocator(10))
ax3.yaxis.set_minor_locator(MultipleLocator(2))
ax3.xaxis.grid(False)
ax3.yaxis.grid(False)
ax3.tick_params(which='major', width=2)
ax3.tick_params(which='major', length=7)
ax3.tick_params(which='minor', length=3)
ax3.legend(loc='lower left')
plt.show()

