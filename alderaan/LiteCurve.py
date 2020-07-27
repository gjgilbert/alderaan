import numpy as np
import astropy
from   astropy.io import fits as pyfits

from .constants import *

__all__ = ['LiteCurve']

class LiteCurve:
    def __init__(self, time=None, flux=None, error=None, cadno=None, quarter=None, channel=None, \
                centroid_col=None, centroid_row=None, mask=None):
        
        self.time         = time
        self.flux         = flux
        self.error        = error
        self.cadno        = cadno
        self.quarter      = quarter
        self.channel      = channel
        self.centroid_col = centroid_col
        self.centroid_row = centroid_row
        self.mask         = mask
        
    
    def to_fits(self, target, filename):
        """
        Save a LiteCurve() object as a fits file

        Parameters
        ----------
            self : LiteCurve() object

            target : string

            filename : string
        """
        # make primary HDU
        primary_hdu = pyfits.PrimaryHDU()

        header = primary_hdu.header

        header['TARGET']  = target

        primary_hdu.header = header

        # add it to HDU list
        hdulist = []
        hdulist.append(primary_hdu)

        hdulist.append(pyfits.ImageHDU(self.time, name='TIME'))
        hdulist.append(pyfits.ImageHDU(self.flux, name='FLUX'))
        hdulist.append(pyfits.ImageHDU(self.error, name='ERROR'))
        hdulist.append(pyfits.ImageHDU(self.cadno, name='CADNO'))
        hdulist.append(pyfits.ImageHDU(self.quarter, name='QUARTER'))
        hdulist.append(pyfits.ImageHDU(self.channel, name='CHANNEL'))
        hdulist.append(pyfits.ImageHDU(self.centroid_col, name='CENT_COL'))
        hdulist.append(pyfits.ImageHDU(self.centroid_row, name='CENT_ROW'))
        hdulist.append(pyfits.ImageHDU(np.array(self.mask, dtype='int'), name='MASK'))

        hdulist = pyfits.HDUList(hdulist)   
        hdulist.writeto(filename, overwrite=True)

        return None