#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:08:20 2018

@author: vganapa1
"""
import matplotlib.pyplot as plt
import numpy as np

from E30_Resolution import NAfilter, filter_function_NA, \
                            create_point, fresnel_prop

Nx = 2**9
Ny = 2**9
wavelength = 600e-9 #[m]
NA = 0.5
dx = 100e-9 #[m]
dy = dx
L = dx*Nx
z = 10e-6 #[m]
crop = 230
z_vec = np.arange(-10,10,1)*(1e-6) #[m]

z_stack = np.zeros([Nx,Ny,len(z_vec)], dtype = np.complex64)

u_0 = create_point(Nx,Ny,0,0)
H_NA = NAfilter(Nx,L,wavelength,NA)
u_f = filter_function_NA(u_0,H_NA,Nx,incoherent=False)

plt.figure()
plt.imshow(np.abs(np.fft.fftshift(H_NA)))


for i,z in enumerate(z_vec):
    u_2 = fresnel_prop(u_f,dx,dy,z,wavelength)

    plt.figure()
    plt.imshow(np.abs(u_2[crop:-crop,crop:-crop])**2, interpolation = 'none')
    plt.title('Image at z = ' + str(z*10**6) + ' microns')   
    
    z_stack[:,:,i] = u_2 

plt.figure()
plt.imshow(np.log(np.abs(z_stack[crop:-crop,Ny//2,:])**2), interpolation = 'none')
plt.title('x-z slice')

plt.figure()
plt.imshow(np.abs(z_stack[crop:-crop,crop:-crop,len(z_vec)//2])**2, interpolation = 'none')
plt.title('x-y slice')