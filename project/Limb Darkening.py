#-*- coding: utf-8
"""
 Program to correct limb darkening in python based on ILD/swwidl
 Based on:
  https://hesperia.gsfc.nasa.gov/ssw/gen/idl/solar/darklimb_correct.pro

 Coefficients taken from 
   Cox, A. N.: Allen's Astrophysical Quantities, Springer, 2000 taken from IDL

"""


import numpy as np
import matplotlib.pyplot as plt
import astropy.io
from astropy.io import fits
import os
import sunpy.map 
import drms
import cv2 as cv
from PIL import Image




def writefits(image):
 	os.system("rm -r limbcorrect.fits")
 	hdu = fits.PrimaryHDU(image)
 	hdul = fits.HDUList([hdu])
 	hdul.writeto('limbcorrect.fits')

def figure(image,title="image",save=False):
 	plt.figure(figsize=(8,8), frameon=False)
 	plt.title('Image')
 	plt.imshow(image,cmap = 'Greys_r',origin='lower')
 	if save == True:
		plt.savefig(title+".png")
 	else:
		pass
 	plt.show()


def darklimb(array):
 	
 	"""
 	  Darklimb function:
 	  
 	  It is requiered the files to be in a .FITS file in order to take
 	  advantage of the Header structure.
 	  
 	  Output:
 	    Dos arrays: The first image is the corrected array, the second
 	    one is the original. 
 	"""
 	
 	# -------------------------------------------------------------
 	
 	def darklimb_u(ll):
		pll = np.array([1.0,ll,ll**2,ll**3,ll**4,ll**5])
		au = -8.9829751
		bu = 0.0069093916
		cu = -1.8144591e-6
		du = 2.2540875e-10
		eu = -1.3389747e-14
		fu = 3.0453572e-19
		a=np.array([au,bu,cu,du,eu,fu])
		ul = sum(a*pll)
		return ul

 	def darklimb_v(ll):
		pll = np.array([1.0,ll,ll**2,ll**3,ll**4,ll**5])
		av = 9.2891180
		bv = -0.0062212632
		cv = 1.5788029e-6
		dv = -1.9359644e-10
		ev = 1.1444469e-14
		fv = -2.599494e-19
		a=np.array([av,bv,cv,dv,ev,fv])
		vl = sum(a*pll)
		return vl
 	
 	# -------------------------------------------------------------
 	
 	# Read data files
 	data = fits.getdata(name,0)
 	head = fits.getheader(name,0)

 	# Parameters needed for the function
 	wavelength = head['WAVELNTH'] # Wavelenght
 	xcen = head['CRPIX1'] # X center
 	ycen = head['CRPIX2'] # Y center
 	radius = head['RSUN_OBS']/head['CDELT1'] # Pixels result
 	size = head['NAXIS1'] # X array size

 	ll =1.0*wavelength

 	array = np.array(data) # Convert data into numpy arrays
 	NaNs = np.isnan(array) # Look for NANs
 	array[NaNs] = 0.0      # Make zero all NANs

 	# Apply correction
 	ul = darklimb_u(ll)
 	vl = darklimb_v(ll)

 	xarr = np.arange(0,size,1.0) # Make X array
 	yarr = np.arange(0,size,1.0) # Make Y array
 	xx, yy = np.meshgrid(xarr, yarr) # Make XY array
 	# z: Make array so that the zero center is the center XY
 	# Performed in order to make a circle
 	z = np.sqrt((xx-xcen)**2 + (yy-ycen)**2) 
 	# grid: Normalize the circle so that inside radius is the unity
 	grid = z/radius 
 	out = np.where(grid>1.0) # Look for the values greater than unity
 	grid[out] = 0.0 # Make zero all those values (Domain of arcsin)

 	limbfilt =  1.0-ul-vl+ul*np.cos(np.arcsin(grid))+vl*np.cos(np.arcsin(grid))**2

 	# Final image
 	imgout=np.array(array/limbfilt)
 	
 	return imgout, array
###############################################################################

# name = 'D:/Desktop/Uni/project/downloads/hmi.ic_nolimbdark_720s.20220203_000000_TAI.3.continuum.fits'
# corrected, original = darklimb(name)
# figure(corrected,title='corrected1',save=True)
# figure(original,title='original1',save=True)

# l_map = sunpy.map.Map(name)
# l_map.peek()

# DRMS series name
# series = 'hmi.Ic_noLimbDark_720s'
# cv.namedWindow('output', cv.WINDOW_NORMAL)
# def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
#     dim = None
#     (h, w) = image.shape[:2]

#     if width is None and height is None:
#         return image
#     if width is None:
#         r = height / float(h)
#         dim = (int(w * r), height)
#     else:
#         r = width / float(w)
#         dim = (width, int(h * r))

#     return cv.resize(image, dim, interpolation=inter)

# k = fits.open(name)
# image_data = np.array(k[1].data)
# image = cv.imread(image_data)
# plt.imshow(image_data)
# plt.savefig('picture.jpeg')
# resize = ResizeWithAspectRatio(image_data, height=500)
# img = resize
# min_val, max_val = img.min(), img.max()
# img = 255.0*(img - min_val)/(max_val - min_val)
# img = img.astype(np.uint8)
# cv.imshow('resize', resize)
# cv.waitKey(0)
# cv.imread('picture.jpeg')




# plt.imsave('butsex.jpeg', image_data)




# Create DRMS JSON client, use debug=True to see the query URLs
# c = drms.Client()

# # Query series info
# si = c.info(series)

# # Print keyword info
# print('Listing keywords for "%s":\n' % si.name)
# for k in sorted(si.keywords.index):
#     ki = si.keywords.loc[k]
#     print(k)
#     print('  type ....... %s ' % ki.type)
#     print('  recscope ... %s ' % ki.recscope)
#     print('  defval ..... %s ' % ki.defval)
#     print('  units ...... %s ' % ki.units)
#     print('  note ....... %s ' % ki.note)
#     print()
































    