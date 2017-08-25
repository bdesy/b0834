from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, ifftshift, ifft2, fftfreq
from mpl_toolkits.axes_grid1 import make_axes_locatable

def binning(array, binx=1, biny=1, use='mean'):
    '''Bins the given 2D array.
       To the mean value of bin if use='mean' and to the sum if use='sum'
       binx, biny  must be integers divisors of array.shape[1] and [0].
       '''
    if use == 'mean':
        return array.reshape(array.shape[0]/biny, biny, array.shape[1]/binx, binx).mean(axis=1).mean(axis=2)
    elif use == 'sum':
        return array.reshape(array.shape[0]/biny, biny, array.shape[1]/binx, binx).sum(axis=1).sum(axis=2)

    
def binning1D(array, nbin=1, use='mean'):
    '''Bins the given 1D array.
       To the mean value of bin if use='mean' and to the sum if use='sum'
       nbin must be an integer divisor of array.shape[0].
       '''
    if use == 'mean':
        return array.reshape(array.shape[1]/nbin, nbin).mean()
    elif use == 'sum':
        return array.reshape(array.shape[1]/nbin, nbin).sum()

    
def aperture00(apex):
    '''Outputs the aperture of a parabola passing by origin with given apex.'''
    return (-apex[1])/((-apex[0])**2)


def parabola(xaxis, apex, aperture):
    '''from a given array of x values, apex and aperture, 
        outputs an array of y values for the parabola
        '''
    return aperture*(xaxis - apex[0])**2 + apex[1]


def get_mask(xaxis, yaxis, apex, fvw=0, fhw=0, lim=True, rect=[0,0], a = None):
    '''Creates an array the size of given axes, value 0 everywhere except 
       value 1 for the parabola passing by origin and given by the apex 
       and the vertical width
       xaxis: 1D array
       yaxis: 1D array
       apex: list or array of shape [x, y, width]
       fvw: fixed vertical width, if desired the same for all masks
       fhw: fixed horizontal width, if desired the same for all masks
       lim: limits the width of the parabola depending of the brightness of arclet (apex[2])
       rect: size of rectangle to hide noise near the origin, dimensions of [mHz, ms] (same as plot)
       '''
    try:
        width = apex[2]
    except IndexError:
        pass
    if fvw == 0:
        y_u = apex[1] + width*1./2
        y_d = apex[1] - width*1./2
    else:
        y_u = apex[1] + fvw*1./2
        y_d = apex[1] - fvw*1./2       
    v_u, v_d = [apex[0], y_u], [apex[0], y_d]
    if a == None:
        a_u, a_d = aperture00(v_u), aperture00(v_d)
    else:
        a_u, a_d = a, a
    X, Y = np.meshgrid(xaxis, yaxis[len(yaxis)/2:,])
    arr = X + 1j*Y
    up = ([a_d*( np.real(arr)- v_d[0] )**2 + v_d[1] <= np.imag(arr) ][0]*1.0)
    down = ([a_u*( np.real(arr)- v_u[0] )**2 + v_u[1] >= np.imag(arr) ][0]*1.0)
    if fhw!=0:
        get_mask.dist = fhw
    else:
        get_mask.dist = width*1500
    if lim:
        limit = ([np.real(arr) >= apex[0]-get_mask.dist ][0]*1.0)*([np.real(arr) <= apex[0]+get_mask.dist][0]*1.0)
        maskh = up * down * limit
    else:
        maskh = up * down
    if rect!=[0,0]:
        vw = rect[1]
        hw = rect[0]
        horiz = ([y_u - vw  <= np.imag(arr) ][0]*1.0)*([y_u  >= np.imag(arr) ][0]*1.0)
        vert = ([apex[0] - hw*1./2 <= np.real(arr) ][0]*1.0)*([apex[0] + hw*1./2 >= np.real(arr) ][0]*1.0)
        othermask = ((horiz*vert)-1.)*-1.
        maskh = maskh*othermask
    return np.vstack(( np.zeros(maskh.shape), maskh))

def get_ss_cs(data, baseline, padding=False, show=False):
    '''From a dynamic spectrum outputs a list of [secondary spectrum, conjugate spectrum, dynamic spectrum used]
    baseline : 257 if single dish data, 258 if complex VLBI data
    padding : adds a contour of zero value around the dynamic spectrum
    '''
    if padding:
        I_pad = np.vstack((data, np.zeros(data.shape)))
        I_pad = np.hstack((I_pad, np.zeros(I_pad.shape)))
        I_pad = np.roll(np.roll(I_pad, data.shape[0]/2, axis=0), data.shape[1]/2, axis=1)
        if show:
            fig = plt.figure(figsize=(11,9))
            plt.imshow(I_pad, cmap='Greys', aspect='auto', origin='lower')
            plt.title('Representation of the padding')
            plt.show()
    else:
        I_pad = data
    if baseline == 257:
        ftI = fftshift(fft2(I_pad))     
        SS = (ftI * np.conj(ftI))

    elif baseline == 258:
        ftI = fftshift(fft2(I_pad))  
        ftIm = np.roll(np.roll(np.fliplr(np.flipud(ftI)), 1, axis=1), 1, axis=0)
        SS = ftI * ftIm
    return [SS, ftI, I_pad]

def perc_error(real, exp, output='str'):
    '''Given a theoretical value and an experimental one, 
    outputs a percentage of error (already multiplied by 100) on the measurement.
    output: 'str' or 'float'
    '''
    val = (np.abs(real*1.-exp*1.)/real*1.)*100.
    if output=='float':
        return val
    else:
        return "%.2f"%val + '%'


def correlate(x, y):
    '''Computes the Pearson's correlation coefficient between two 2D matrices'''
    return np.sum(  np.absolute(x-np.mean(x)) * np.absolute(y-np.mean(y))  ) \
                                /(np.sqrt(  np.sum(np.absolute(x-np.mean(x))**2)*np.sum(np.absolute(y-np.mean(y))**2))  )

def plot_two(arr1, arr2, extent1, extent2, inter=None, vmin1=None, vmax1=None, title1='',  vmin2=None, vmax2=None, title2='',cmap=None, xlabel='x', ylabel='y',):
    fig = plt.figure(figsize=(20,9))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

    im1 = ax1.imshow(arr1, 
              aspect="auto",
              origin='lower',
              interpolation=inter, 
              cmap=cmap,
              vmin=vmin1, vmax=vmax1,
              extent=extent1)
    ax1.set_title(title1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)

    im2 = ax2.imshow(arr2, 
              aspect="auto",
              origin='lower',
              interpolation=inter, 
              cmap=cmap,
              vmin=vmin2, vmax=vmax2,
              extent=extent2)
    ax2.set_title(title2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    plt.show()
    
def get_theta(dn,thetaR,thetaT,beta,s):
    theta = [None,None]
    p = [4.,
         2.*thetaR-8.*beta,
         4.*beta**2.-4.*beta*thetaR,
         2.*beta**2.*thetaR,
         0,
         0,
         -(s*dn*thetaR*thetaT)**2.]
    if dn<0:
        if beta >0:
            theta=np.real(np.sort(np.roots(p)[(np.roots(p)>0)&(np.imag(np.roots(p))==0)])[:2])
        else: 
            theta = np.array([np.NaN,beta])
    elif dn>0:
        theta = np.real(np.sort(np.roots(p)[np.roots(p)>=0])[-1])
        if beta < 0:
            theta = np.append(theta,beta)
        else:
            theta = np.append(theta,np.NaN)
    return theta

def get_mu(dn,thetaR,thetaT,beta,s):
    theta = get_theta(dn,thetaR,thetaT,beta,s)
    mu = (1. / (1. - (s * dn * thetaR**2. * thetaT)/( 8. * theta**4. * (1. + thetaR/(2. * theta))**(3./2.)) 
                + ( s * dn * thetaR * thetaT)/(theta**3.*(1. + thetaR/(2. * theta))**(1./2.))))
    for i in range(len(mu)): #This isn't working properly
        if theta[i] == beta:
            mu[i] = 1.
    return np.real(mu)

def get_soln(dn,thetaR,thetaT,beta):
    theta = get_theta(dn,thetaR,thetaT,beta,s)
    mu = (1./
          (1. - ((3.37240597513748e32 * dn * thetaR**2. * thetaT)/(theta**4. * ( 1. + 1.0313531353135315e8 * thetaR / theta)**(3./2.)))
           + ((1.3079539333842376e25 * dn * thetaR * thetaT)/(theta**3.*(1. + 1.0313531353135315e8 * thetaR / theta)**(1./2.)))))
    for i in range(len(mu)): #This isn't working properly
        if theta[i] == beta:
            mu[i] = 1.
    return theta,mu
