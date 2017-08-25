#!/usr/bin/python
''' 
Computes the flux of a given secondary spectrum (SS), using DS that have been :
        - averaged over gates 0, 1 and 6 (from Dana's directories)
        - normalized to the mean and std of Ar-GBT spectrum (method at start of Notebook 16_other_frequency_bands)
        - gaps in data have been set to 0
        - padded all around with 0
        - normalized for change of frequency using script normalize_with_lambda.py

Baseline 257 is Arecibo single dish data, 258 is VLBI AR-GBT data
Step between doppler shift coordinates [fD_start, fD_end] determines the precision and range desired for fluxes
horizontal/vertical width (hw, vw) to define the size of the arclet

output : saves a numpy array with the values of fluxes and their doppler frequency coordinate
         to load use flurcurve = numpy.load(PATH).reshape(2, -1)
             fluxcurve[0] will be the doppler shift coordinates array and 
             fluxcurve[1] the fluxes values
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift, fftfreq
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, sys
from custom_functions import *
from tqdm import tqdm

__author__ = 'Beatrice Desy'

# Setting parameters

center_freq = 314.5e6                # central frequency of observations, here all DS have been normalized to 314.5MHz
fD_start, fD_end = -44, 44           # values of doppler frequency where the 'swiping' along parabola will happen,
                                     # -44 to 44 covers the whole SS
step = 0.01             # step between doppler shift coordinates, 0.1 is good for testing, 0.01 for measuring
hw =  3.               # in mHz, distance between apex and one end of the arclet
vw =  0.001            # maximal vertical width of arclet at the apex
inpath = '/mnt/scratch-lustre/bdesy/b0834/data/normalized_data/correct_channels/'
outpath = '' 

# Parameters in console to do quickly all frequency bands without playing here in the code
# the terminal command should look like "python get_flux.py 257 00" to directly choose the baseline and frequency band
# but the scripts handle just the "python get_flux.py" call and will ask for other info

try:
    baseline = int(sys.argv[1])
except IndexError:
    baseline = int(raw_input('baseline ? (257 or 258) '))

try:
    freqband = sys.argv[2]
except IndexError:
    freqband = raw_input('frequency band? (00, 01, 02 or 03) ')

# Recovering the correct DS, might need changing depending where things are    

ds = inpath + str(baseline) + 'freq' + freqband + 'norm_lambda03-08.npy'

data = np.load(ds) 

# Defining the main function

def get_flux(SS, x, a, xaxis, yaxis, hw, vw, show=False):
    '''
    Computes the flux of a given point on the main parabola of aperture a
    SS : secondary spectrum
    x  : doppler shift coordinate of the apex where the flu will be computed
    a  ; aperture of parabola
    xaxis, yaxis : axes of SS
    hw, vw : limits on the arclet's shape
    show : if desired display on the arclet on the SS, mainly for use outside of script
    '''
    if x == 0.:
        return np.absolute(SS[SS.shape[0]/2, SS.shape[1]/2])
    y = a * x**2
    X, Y = np.meshgrid(xaxis, yaxis[len(yaxis)/2:,])
    arr = X + 1j*Y

    y_u = y + vw*1./2
    y_d = y - vw*1./2
    v_u, v_d = [x, y_u], [x, y_d]
    up = ([-a*( np.real(arr)- v_d[0] )**2 + v_d[1] <= np.imag(arr) ][0]*1.0)
    down = ([-a*( np.real(arr)- v_u[0] )**2 + v_u[1] >= np.imag(arr) ][0]*1.0)
    limit = ([np.real(arr) >= x-hw ][0]*1.0)*([np.real(arr) <= x+hw][0]*1.0)
    maskh = up * down * limit
       
    mask = np.vstack(( np.zeros(maskh.shape), maskh))
    if show:
        extent=[doppler[0], doppler[-1], delay[len(delay)/2], delay[-1]]
        fig = plt.figure(figsize=((12,12)))
        pltss = binning(np.log10(np.absolute(SS)), 2, 16)
        plt.imshow(pltss[pltss.shape[0]/2:, :], origin='lower', aspect='auto', interpolation=None, cmap='Greys',
              extent=extent)
        plt.colorbar()
        plt.imshow((np.ma.masked_where(mask == 0, mask))[mask.shape[0]/2:, :], 
                       aspect="auto", origin="lower", cmap="winter",
                       extent=extent, alpha=0.7, interpolation='spline16')
        plt.show()
    return np.sum(np.absolute(mask*SS))

a = (5.577e13)/(center_freq)**2  # From Brisken's paper

xes = np.arange(fD_start, fD_end, step)
fluxes1 = np.zeros(xes.shape)

SS = binning(get_ss_cs(data, baseline)[0], 2, 2)
SS[:SS.shape[0]/2] = 0
SS[:, SS.shape[1]/2] = 0
SS[:SS.shape[0]/2 + 100, SS.shape[1]*29/60:SS.shape[1]*31/60] = 0 #to avoid inifities near origin, could be bigger

freq = (np.arange(SS.shape[0])*8./SS.shape[0]+310.5) 
time = np.arange(SS.shape[1])*6729./SS.shape[1]
doppler = (fftshift(fftfreq(SS.shape[1]))*1./ time[1])*1e3
delay = (np.arange(SS.shape[0])-SS.shape[0]/2)*1./(8.0e6)*1e3
delayh = delay[len(delay)/2:]
  
for i in tqdm(range(xes.shape[0])):
    fluxes1[i] = get_flux(SS, xes[i], a, doppler, delay, hw=hw, vw=vw)

output = np.vstack((xes, fluxes1))    # to output the values of fluxes and their doppler frequency coordinate in one array

np.save(outpath + str(baseline) + 'freq' + freqband + 'flux_norm_lambda' + str(int(hw)) + 'x' + str(vw) + '.npy', output)
