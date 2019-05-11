#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:04:41 2018

@author: vganapa1
"""

import numpy as np
import os
import glob
import imageio

### Inputs that can be changed ###
input_folder = 'Flatworm_SampleNumber0002_RegionNumber0001'
num_GPUs = 4
###############################


### Functions ###
def get_background_reference(input_folder_name):
    filenames = glob.glob(input_folder_name + "/*.png")
    img_ave = 0
    for f in filenames:
        img = imageio.imread(f)
        img_ave = img_ave + np.sum(img)/float(img.shape[0]*img.shape[1])
    img_ave = img_ave/float(len(filenames))
    return img_ave
###############################


total_length = 2048 # assumes square image
patch_size = 512
overlap = 64

### Find background_threshold from the black images
black_folder = input_folder + '/BlackReference'
background_threshold = get_background_reference(black_folder)

print('Background threshold is: ', background_threshold)

x_vec = np.arange(0,total_length,patch_size-overlap)

residual = total_length - x_vec[-1]
if residual < patch_size:
    x_vec = x_vec[:-1]

file_name = input_folder + '_script.sh'

with open(file_name,'w') as f:

    count = 0

    for x in x_vec:
        for y in x_vec:
    #        print([x,y])
            
            command = 'python FP_Low_Res_Reconstruction_v2_3D.py --input ' + input_folder + ' --output patch_' + str(count)+ \
                  ' -g '+ str(num_GPUs) + ' --np ' + str(patch_size) + ' --nstart_x ' + str(x) + \
                   ' --nstart_y ' + str(y) + ' --background ' + str(background_threshold)
            print(command)
            f.write(command + " \n")
            
            count += 1

os.system("chmod +x " + file_name)

