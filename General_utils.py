#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:49:15 2022

@author: katie-lou
"""
from pathlib import Path
import numpy as np
import pandas as pd

root_directory = "/Users/katie-lou/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/"


#%% Color arrays and cmaps consistent with design diagrams in Powerpoint
from matplotlib.colors import ListedColormap
N = 256

# Can use colour grid to determine colormaps. Haven't replaced the code below yet as it's not necessary for function
biofiltercols = np.zeros((9,3))
biofiltercols[0] = [0   , 0     , 0] # Use Black instead of White!
biofiltercols[1] = [195 , 41  , 246]
biofiltercols[2] = [0   , 0   , 256]
biofiltercols[3] = [117 , 255 , 202]
biofiltercols[4] = [119 , 255 , 77]
biofiltercols[5] = [196 , 253 , 80]
biofiltercols[6] = [255 , 191 , 72]
biofiltercols[7] = [255 , 120 , 48]
biofiltercols[8] = [139 , 69  , 19]
biofiltercols    = biofiltercols/256


cmap1 = np.ones((N, 4))
cmap1[:, 0] = np.linspace(0, 1, N)
cmap1[:, 1] = np.linspace(0, 1, N)
cmap1[:, 2] = np.linspace(0, 1, N)
cmap1 = ListedColormap(cmap1)

cmap2 = np.ones((N, 4))
cmap2[:, 0] = np.linspace(0, 195/256, N)
cmap2[:, 1] = np.linspace(0, 41/256, N)
cmap2[:, 2] = np.linspace(0, 246/256, N)
cmap2 = ListedColormap(cmap2)

cmap3 = np.ones((N, 4))
cmap3[:, 0] = np.linspace(0, 0, N)
cmap3[:, 1] = np.linspace(0, 0, N)
cmap3[:, 2] = np.linspace(0, 1, N)
cmap3 = ListedColormap(cmap3)

cmap4 = np.ones((N, 4))
cmap4[:, 0] = np.linspace(0, 117/256, N)
cmap4[:, 1] = np.linspace(0, 251/256, N)
cmap4[:, 2] = np.linspace(0, 202/256, N)
cmap4 = ListedColormap(cmap4)

cmap5 = np.ones((N, 4))
cmap5[:, 0] = np.linspace(0, 119/256, N)
cmap5[:, 1] = np.linspace(0, 255/256, N)
cmap5[:, 2] = np.linspace(0, 77/256, N)
cmap5 = ListedColormap(cmap5)

cmap6 = np.ones((N, 4))
cmap6[:, 0] = np.linspace(0, 196/256, N)
cmap6[:, 1] = np.linspace(0, 253/256, N)
cmap6[:, 2] = np.linspace(0, 80/256, N)
cmap6 = ListedColormap(cmap6)

cmap7 = np.ones((N, 4))
cmap7[:, 0] = np.linspace(0, 255/256, N)
cmap7[:, 1] = np.linspace(0, 191/256, N)
cmap7[:, 2] = np.linspace(0, 72/256, N)
cmap7 = ListedColormap(cmap7)

cmap8 = np.ones((N, 4))
cmap8[:, 0] = np.linspace(0, 255/256, N)
cmap8[:, 1] = np.linspace(0, 73/256, N)
cmap8[:, 2] = np.linspace(0, 48/256, N)
cmap8 = ListedColormap(cmap8)

cmap9 = np.ones((N, 4))
cmap9[:, 0] = np.linspace(0, 255/256, N)
cmap9[:, 1] = np.linspace(0, 49/256, N)
cmap9[:, 2] = np.linspace(0, 32/256, N)
cmap9 = ListedColormap(cmap9)

# List them for looped recall:
biofiltercmaps = [cmap1,cmap2,cmap3,cmap4,cmap5,cmap6,cmap7,cmap8,cmap9]

#%% Color arrays of Macbeth Color Chart

macbethcols = np.zeros((20,3))

# Colors found using Apple Digital Colour Meter and Google Images
macbethcols[0] = [167   ,   63  ,   11  ] # Darkskin
macbethcols[1] = [253   ,   145 ,   147 ] # Lightskin
macbethcols[2] = [99    ,   129 ,   214 ] # Blue Sky
macbethcols[3] = [92    ,   216 ,   4   ] # Foliage
macbethcols[4] = [164   ,   126 ,   233 ] # Blue Flower
macbethcols[5] = [136   ,   231 ,   223 ] # Bluish Green
macbethcols[6] = [252   ,   140 ,   10  ] # Orange
macbethcols[7] = [37    ,   72  ,   255 ] # Purplish Blue
macbethcols[8] = [252   ,   62  ,   100 ] # Moderate Red
macbethcols[9] = [94    ,   0   ,   144 ] # Purple
macbethcols[10] = [226  ,   255 ,   31  ] # Yellow Green
macbethcols[11] = [254  ,   194 ,   10  ] # Orange Yellow
macbethcols[12] = [0    ,   0   ,   255 ] # Blue
macbethcols[13] = [92   ,   197 ,   44  ] # Green
macbethcols[14] = [255  ,   0   ,   0   ] # Red
macbethcols[15] = [255  ,   223 ,   12  ] # Yellow: Moderated with some blue to be more visible
macbethcols[16] = [250  ,   62  ,   205 ] # Magenta
macbethcols[17] = [24   ,   168 ,   245 ] # Cyan
macbethcols[18] = [150  ,   150 ,   150 ] # White: Use Gray to make visible
macbethcols[19] = [0    ,   0   ,   0   ] # Black
macbethcols    = macbethcols/256

macbethnames= np.array(["darkskin",
                        "lightskin",
                        "bluesky",
                        "foliage",
                        "blueflower",
                        "bluishgreen",
                        "orange",
                        "purplered",
                        "moderatered",
                        "purple",
                        "yellowgreen",
                        "orangeyellow",
                        "blue",
                        "green",
                        "red",
                        "yellow",
                        "magenta",
                        "cyan",
                        "white",
                        "black"])

macbethdict = {macbethnames[i]: macbethcols[i] for i in range(len(macbethnames))}

#%% Wavelength to RGB
def wavelength_to_rgb(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    elif wavelength >= 750 and wavelength <= 900: # Added in fade to grey in NIR
        R = (1-(900-wavelength)/(900-750))**gamma
        G = (1-(900-wavelength)/(900-750))**gamma
        B = (1-(900-wavelength)/(900-750))**gamma
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B)

#%%% For spectra overall RGB output
def pdwaveform_to_rgb(waveform):
    '''Sample spectral heights across the spectrum, assuming relatively broad peaks.
    Must input waveform as a pandas dataframe with the wavelength as the index
    '''
    from scipy.integrate import trapezoid
    data = waveform
    data = data.rolling(5).mean()
    data = ((data-data.min())/(data.max())).dropna()
    n = len(data)

    R_tot = 0
    B_tot = 0
    G_tot = 0

    if n>0: # n = 0 when whole dataframe is NaNs, at which point just plot black!
        for wavelength in data.index:
            scalar = data[wavelength]
            wavelength = float(wavelength)

            if wavelength >= 400 and wavelength <= 440:
                attenuation = 0.3 + 0.7 * (wavelength - 400) / (440 - 300)
                R = ((-(wavelength - 440) / (440 - 300)) * attenuation)
                G = 0.0
                B = (1.0 * attenuation)
            elif wavelength >= 440 and wavelength <= 490:
                R = 0.0
                G = ((wavelength - 440) / (490 - 440))
                B = 1.0
            elif wavelength >= 490 and wavelength <= 510:
                R = 0.0
                G = 1.0
                B = (-(wavelength - 510) / (510 - 490))
            elif wavelength >= 510 and wavelength <= 580:
                R = ((wavelength - 510) / (580 - 510))
                G = 1.0
                B = 0.0
            elif wavelength >= 580 and wavelength <= 645:
                R = 1.0
                G = (-(wavelength - 645) / (645 - 580))
                B = 0.0
            elif wavelength >= 645 and wavelength <= 750:
                attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
                R = (1.0 * attenuation)
                G = 0.0
                B = 0.0
            else:
                R = 0.0
                G = 0.0
                B = 0.0
            R = R * scalar
            G = G * scalar
            B = B * scalar

            R_tot = R_tot + R
            G_tot = G_tot + G
            B_tot = B_tot + B

        scaledown = trapezoid(data)
        R_tot = R_tot / scaledown
        G_tot = G_tot / scaledown
        B_tot = B_tot / scaledown

    return (R_tot, G_tot, B_tot)


# Waveform to RGB more generalised
def spec2rgb(wavs, vals):
    # Just a weighted mean of contributions at each wavelength
    if len(wavs)!=len(vals):
        print("Ensure wavelengths and values arrays are the same length!")
    else:
        weights = vals/np.sum(vals)
        rgbvals = np.array([wavelength_to_rgb(w) for w in wavs])
        specrgb = np.average(rgbvals,weights=weights,axis=0)
        return(tuple(specrgb))


def setup_directories(directory_name_array):
    import os
    for directory_name in directory_name_array:
        if os.path.exists(directory_name) == False:
            os.makedirs(directory_name)

def lorentzian(x, x0, a, gam, b):
    return b + a * 0.5 * gam ** 2 / ((0.5 * gam) ** 2 + (x - x0) ** 2)
def gaussian(x, mu, sig,normalise="area"):
    if normalise == "area":
        gaussian = np.exp(-np.power((x - mu)/sig, 2.)/2)
        gaussian = gaussian/np.sum(gaussian)
    elif normalise == "max":
        gaussian = np.exp(-np.power((x - mu) / sig, 2.) / 2)
        gaussian = gaussian/gaussian.max()
    else:
        gaussian = np.exp(-np.power((x - mu) / sig, 2.) / 2)
    return gaussian
def rolling_mean_smoothing1d(array_to_smooth,window_width=10,axis=0):
    from scipy import ndimage
    kernel = np.ones(window_width) / window_width
    smootharray = ndimage.convolve1d(array_to_smooth, kernel, axis=axis)
    return smootharray
def norm_by_percentile(array,percentile,nonneg=True):
    normarray = array/np.nanpercentile(array,percentile)
    normarray[normarray>1]=1
    if nonneg == True:
        normarray[normarray<0]=0
    return normarray

def namefilter(string, substr):
    # Useful when selecting files based on acquisition conditions listed in filename
    return [str for str in string if all(sub in str for sub in substr)]
