#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:54:58 2018

@author: vganapa1
"""

import numpy as np
import time
from ShiftAddFunctions import get_derived_params_SA, \
                              shift_add
import matplotlib.pyplot as plt
from FP_Reconstruction_Functions_3D import read_images
import sys


##############
### INPUTS ###

training_data_folder = 'Flatworm_SampleNumber0002_RegionNumber0001'
nstart = [0,0]
background_removal = True
threshold = 105.0
num_stacks = 1
##############

# wavelength of illumination in microns, assume monochromatic
wavelength = 0.518 

# numerical aperture of the objective
NA = 0.3

# magnification of the system
mag = 9.24

#6.5um pixel size on the sensor plane
dpix_c = 6.5

# number of pixels at the output image patch
Np = np.array([2048,2048])

# center of image
ncent = np.array([1024,1024])


### LED array geometries ###

# spacing between neighboring LEDs, 4mm
ds_led = 4e3 

# distance from the LED to the object
z_led = 69.5e3

# diameter of number of LEDs used in the experiment
dia_led = 9.0

# center LED
# h: horizontal, v: vertical

lit_cenv = 15
lit_cenh = 16


# refocusing parameter: units um
# refocusing step size
dz = 250.0 #250

# refocusing range: units um
zmin = -500.0 #-1000.0 
zmax = 750.0 # 500 #1000.0


##############
##############
N_patch_center = np.array([1024,1024])
N_img_center = np.array([1024,1024])

u, v, Nz, z_vec, Nimg, Tanh_lit, Tanv_lit = get_derived_params_SA\
                       (NA, wavelength, mag, dpix_c, Np, N_patch_center, N_img_center, \
                        lit_cenv, lit_cenh, ds_led, z_led, zmin, zmax, dz, dia_led)

img_stack = read_images(training_data_folder, Np[0], nstart, background_removal, threshold, num_stacks)
img_stack = img_stack.astype(dtype=np.complex64)



start_time = time.time()
    
# shift-and-add 
tot_mat = shift_add(img_stack, Np, Nz, z_vec, Nimg, Tanh_lit, Tanv_lit, \
              u, v, all_mats=False)

end_time = time.time()
total_time = end_time - start_time
print('Shift and add took', total_time, 'seconds.')

for m in range(0, Nz):
    plt.figure()
    plt.imshow(tot_mat[m,:,:], vmax=300000)
    plt.colorbar()
    plt.show()
    
np.save(training_data_folder + '/tot_mat.npy', tot_mat)
np.save(training_data_folder + '/dz.npy', dz)
np.save(training_data_folder + '/z_vec.npy', z_vec)