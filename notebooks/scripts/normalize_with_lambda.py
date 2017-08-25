#!/usr/bin/python
'''
Corrected script to padd and correct dynamic spectra to the same center frequency
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift, fftfreq
import os
from tqdm import tqdm

__author__ = 'Beatrice Desy'

# Parameters in console to do quickly all frequency bands without playing here in the code
# the terminal command should look like "python normalize_with_lambda.py 257 00" to directly choose the baseline and frequency band
# but the scripts handle just the "python get_flux.py" call and will ask for other info

try:
    baseline = int(sys.argv[1])
except IndexError:
    baseline = int(raw_input('baseline ? (257 or 258) '))
try:
    freq = sys.argv[2]
except IndexError:
    freq = raw_input('frequency band? (00, 01, 02 or 03) ')

# Parameters common across frequency bands   

outpath = '/mnt/scratch-lustre/bdesy/b0834/data/normalized_data/correct_channels/'
inpath = '/mnt/scratch-lustre/bdesy/b0834/data/normalized_data/ds' 
f0 = 314.5                  #  frequency we want the dynamic spectrum to be corrected to

# input could be any file of DS 16384x660, but here the method calls for a .npy format
inds = np.load(inpath + str(baseline) + 'freq' + freq + '.npy')
num_rows=inds.shape[0]
num_columns=inds.shape[1]

i = int(freq[1])

starts = [310.5, 318.5, 326.5, 334.5]
ends = [318.5, 326.5, 334.5, 342.5]

start = starts[i]
end = ends[i]

def correct_channels(Ip2, f0, start, end):
    I_pad = np.pad(Ip2, ((0,num_rows), (0,num_columns)), 'constant', constant_values=0)
    I = np.roll( np.roll( I_pad,num_rows/2,axis=0 ),num_columns/2,axis=1 )
    frequency = np.arange(start, end,8.0/num_rows)   # frequency of every channel
    frequency = frequency + np.diff(frequency)[0]    # make sure we are using the central freq        
    Iout = np.zeros(I.shape,dtype=complex)
    for i in tqdm(range(len(frequency))):
        x = I[i+num_rows/2,:]
        N = x.shape[0]
        n = np.arange(N) - N/2
        k = n.reshape((N,1))*frequency[i]/f0
        M = np.exp(-2.0j*np.pi*k*n/N)
        y = np.dot(M,x)
        # Slow FT back
        k = n.reshape((N,1))
        M = np.exp(2.0j*np.pi*k*n/N)
        Iout[i+num_rows/2,:] = 1./N*np.dot(M,y)
    return Iout

outds = correct_channels(inds, f0, start, end)
np.save(outpath + str(baseline) + 'freq' +  freq + 'norm_lambda.npy', outds)