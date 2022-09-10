import os 
import matplotlib.pyplot as plt 
import astropy.time
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, SqrtStretch
import sunpy.map
from sunpy.net import Fido 
from sunpy.net import jsoc, fido_factory, attrs as a
import drms
from astropy.io import fits 
from astropy.wcs import WCS 
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface

# in_time = aiamap.date

#query full frame HMI image
t0 = astropy.time.Time('2021-12-24T20:34:07', scale='utc', format='isot')
q = Fido.search(
    a.Instrument.hmi,
    a.Physobs.intensity,
    a.Time(t0, t0 + 10*u.s)
    )

m = sunpy.map.Map(Fido.fetch(q))

# create submap from this image, and crop to active region 
m_cutout = m.submap(
    SkyCoord(-699*u.arcsec, -221*u.arcsec, frame=m.coordinate_frame),
    top_right=SkyCoord(-580*u.arcsec, -300*u.arcsec, frame=m.coordinate_frame)
    )

m_cutout.peek()

cutout = a.jsoc.Cutout(
    m_cutout.bottom_left_coord,
    top_right=m_cutout.top_right_coord,
    tracking=True
)

# top_right = SkyCoord(580 * u.arcsec, 300 * u.arcsec, frame=m.coordinate_frame)
# bottom_left = SkyCoord(699 * u.arcsec, 221 * u.arcsec, frame=m.coordinate_frame)
# m_cutout = m.submap(bottom_left, top_right=top_right)

# cutout = a.jsoc.Cutout(
#     m_cutout
# )

c = drms.Client()
series_name = 'hmi.Ic_noLimbDark_720s'
jsoc_email = 'leiftinwell@rocketmail.com'
segment_name = 'continuum'

#construct query with cutout component and sample rate

response = Fido.search(
    a.Time(m_cutout.date - 6*u.h, m_cutout.date + 6*u.h),
    a.jsoc.Series(series_name),
    a.jsoc.Notify(jsoc_email),
    a.jsoc.Segment(segment_name),
    a.Sample(9*u.h),
    cutout,
)

print(response)


#submit, search and download
files = Fido.fetch(response, progress=True)
files.sort()


# # create a map sequence from them 
m_seq = sunpy.map.Map(files, sequence=True)
m_seq.plot()

for m in m_seq:
    m.plot_settings['norm'] = ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch())
m_seq.peek()

plt.show()
















