#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:04:59 2018

@author: vganapa1
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

def F(mat2D): # Fourier tranform centered at zero
    mat2D = np.fft.fftshift(mat2D)
    mat2D = np.fft.fft2(mat2D)
    mat2D = np.fft.fftshift(mat2D)
    return mat2D

def Ft(mat2D): # Inverse Fourier transform centered at zero
    mat2D = np.fft.fftshift(mat2D)
    mat2D = np.fft.ifft2(mat2D)
    mat2D = np.fft.fftshift(mat2D)
    return mat2D

# Load Boston University data

data = loadmat('CLEAN_tutorial/3C111MAY14.mat')
u = data['u']
v = data['v']
amp = data['amp']
phase = data['phase']

# Plot uv data
plt.figure()
plt.scatter(u,v,c=amp)

amp = np.squeeze(amp)
phase = np.squeeze(phase)

# Convert uv data to numpy matrix
num_points = 2**11
max_val = np.max([np.max(u),np.max(v)])
min_val = np.min([np.min(u),np.min(v)])

u_vec = np.linspace(min_val,max_val,num_points)

uv_mat = np.zeros([num_points,num_points],\
                  dtype=np.complex64)
PSF = np.zeros([num_points,num_points],\
               dtype=np.complex64)

for point in range(len(u)):
    
    # find closest point in uv_mat
    u_coord = np.argmin(np.abs(u_vec - u[point]))
    v_coord = np.argmin(np.abs(u_vec - v[point]))

    uv_mat[u_coord,v_coord] = amp[point]*\
                              np.exp(1j*phase[point])
    PSF[u_coord,v_coord] = 1.


plt.figure()
plt.scatter(np.nonzero(np.abs(PSF))[0],\
            np.nonzero(np.abs(PSF))[1], \
            c = np.abs(uv_mat[np.nonzero(np.abs(PSF))]))

PSF = Ft(PSF)
dirty_beam = Ft(uv_mat)

plt.figure()
plt.imshow(np.abs(dirty_beam)**2, vmax=0.6e-9)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(PSF),vmax = 1e-5)
plt.colorbar()

# CLEAN algorithm

noise_floor = 0.5*np.max(np.abs(dirty_beam))
max_iter = 1e2
cleaned_img = np.zeros([num_points,num_points], dtype=np.complex64)


# while loop

max_inds = np.unravel_index(np.argmax(np.abs(dirty_beam)), \
                 dirty_beam.shape)
gain = dirty_beam[max_inds]/np.max(np.abs(PSF))
shifted_real = shift(np.real(PSF),[max_inds[0]-num_points//2,\
                                  max_inds[1]-num_points//2])
shifted_imag = shift(np.imag(PSF), [max_inds[0]-num_points//2,\
                                   max_inds[1]-num_points//2])
dirty_beam = dirty_beam - gain*(shifted_real + 1j*shifted_imag)
cleaned_img[max_inds] = gain

# Filter cleaned_img with gaussian kernel at end
