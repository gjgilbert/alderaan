__all__ = ["LiteCurve"]


import astropy
from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from .constants import *


class LiteCurve:
    def __init__(
        self,
        time=None,
        flux=None,
        error=None,
        cadno=None,
        quarter=None,
        season=None,
        quality=None,
        mask=None,
    ):
        self.time = time
        self.flux = flux
        self.error = error
        self.cadno = cadno
        self.quarter = quarter
        self.season = season
        self.quality = quality
        self.mask = mask

    def clip_outliers(self, kernel_size, sigma_upper, sigma_lower, mask=None):
        """
        Sigma-clip outliers using astropy.stats.sigma_clip() and a median filtered trend
        Applies an additional iteration wrapper to allow for masked cadences

        Parameters
        ----------
            kernel_size : int
                size of the median filter kernel
            sigma_upper : float
                upper sigmga clipping threshold
            sigma_lower : float
                lower sigma clipping threshold
            mask : array-like, bool (optional)
                do not reject cadences within masked regions; useful for protecting transits
        """
        if mask is None:
            mask = np.zeros(len(self.time), dtype="bool")

        loop = True
        count = 0

        while loop:
            # first pass: try median filter
            if loop:
                smoothed = sig.medfilt(self.flux, kernel_size=kernel_size)

                bad = astropy.stats.sigma_clip(
                    self.flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=astropy.stats.mad_std,
                ).mask
                bad = bad * ~mask

            # second pass: try savgol filter (if over 1% of points were flagged with median filter)
            if np.sum(bad) / len(bad) > 0.01:
                smoothed = sig.savgol_filter(
                    self.flux, window_length=kernel_size, polyorder=2
                )

                bad = astropy.stats.sigma_clip(
                    self.flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=astropy.stats.mad_std,
                ).mask
                bad = bad * ~mask

            # third pass: skip outlier rejection
            if np.sum(bad) / len(bad) > 0.01:
                bad = np.zeros_like(mask)

            for k in self.__dict__.keys():
                if type(self.__dict__[k]) is np.ndarray:
                    self.__setattr__(k, self.__dict__[k][~bad])

            mask = mask[~bad]

            if np.sum(bad) == 0:
                loop = False
            else:
                count += 1

            if count >= 3:
                loop = False

        return self

    def remove_flagged_cadences(self, qmask):
        """
        Remove user-specified cadences using a pre-generated mask

        Parameters
        ----------
        qmask : array-like
            boolean array; 1=good, 0=bad
        """
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][qmask])

        return self

    def plot(self):
        """
        Plot the photometry
        """
        fig = plt.figure(figsize=(20, 4))
        plt.plot(self.time, self.flux, "k", lw=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Time [BKJD]", fontsize=24)
        plt.ylabel("Flux", fontsize=24)
        plt.xlim(self.time.min(), self.time.max())

        return fig

    def to_fits(self, target, filename, cadence):
        """
        Save LiteCurve object as a fits file

        Parameters
        ----------
        target : string
            name of target
        filename : string
            where to save the fits file
        """
        # make primary HDU
        primary_hdu = pyfits.PrimaryHDU()

        header = primary_hdu.header

        header["TARGET"] = target
        header["CADENCE"] = cadence

        primary_hdu.header = header

        # add it to HDU list
        hdulist = []
        hdulist.append(primary_hdu)

        hdulist.append(pyfits.ImageHDU(self.time, name="TIME"))
        hdulist.append(pyfits.ImageHDU(self.flux, name="FLUX"))
        hdulist.append(pyfits.ImageHDU(self.error, name="ERROR"))
        hdulist.append(pyfits.ImageHDU(self.cadno, name="CADNO"))
        hdulist.append(pyfits.ImageHDU(self.quarter, name="QUARTER"))

        hdulist = pyfits.HDUList(hdulist)
        hdulist.writeto(filename, overwrite=True)

        return None
