#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:41:20 2018

@author: vganapa1
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from FP_Reconstruction_Functions import read_images, read_images_single_stack
import os, sys
from VisualizerMultiPatchFunctions import load_final_var, CreateFig, save_pngs_patch
import glob
    
### Command Line Inputs

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('-d', type=str, action='store', dest='dataset_folder', \
                        help='main folder containing the subfolders')

parser.add_argument('--count', type=int, action='store', dest='total_patches', \
                    help='Number of total patches', \
                    default = 16)    

parser.add_argument('--overlap', type=int, action='store', dest='overlap', \
                    help='overlap of the patches', \
                    default = 64) 

parser.add_argument('--uf', type=int, action='store', dest='upsample_factor', \
                    help='N_obj/Np, the upsample factor to go from low res to high res', \
                    default = 2) 

parser.add_argument('--ps', type=int, action='store', dest='patch_size', \
                    help='Np, the size of a patch', \
                    default = 512) 

parser.add_argument('--nx0', type=int, action='store', dest='nstart_x_0', \
                    help='Initial starting coordinate for defining region of interest, row coordinate', \
                    default = 0)

parser.add_argument('--ny0', type=int, action='store', dest='nstart_y_0', \
                    help='Initial starting coordinate for defining region of interest, column coordinate', \
                    default = 0)   

parser.add_argument('--ns', type=int, action='store', dest='num_stacks', \
                    help='num_stacks', \
                    default = 1)       

parser.add_argument('--sp', type=int, action='store', dest='save_patches', \
                    help='save_patches', \
                    default = 4)    


parser.add_argument('--zi', type=int, action='store', dest='z_ind', \
                    help='z_ind', \
                    default = 2)  

args = parser.parse_args()

### Inputs from command line args
training_data_folder = args.dataset_folder 
total_patches = args.total_patches
overlap = args.overlap
upsample_factor = args.upsample_factor
patch_size = args.patch_size
nstart_x_0 = args.nstart_x_0
nstart_y_0 = args.nstart_y_0
num_stacks = args.num_stacks
save_patches = args.save_patches
z_ind = args.z_ind

### read in final object data from each patch
all_high_res_guess_1 = []
for i in range(total_patches):
    output_folder = 'patch_' + str(i)
    folder_name = training_data_folder + '/' + output_folder
    iter_vec = np.load(folder_name + '/iter_vec.npy')
    high_res_guess_1 = load_final_var('high_res_guess',folder_name, iter_vec)
    all_high_res_guess_1.append(high_res_guess_1[z_ind,:,:])


### merge together all columns for each row    
x_strips = []
x_patches_vec = np.arange(np.sqrt(total_patches))

count = 0

indices = np.expand_dims(np.arange(0,overlap*upsample_factor)/float(overlap*upsample_factor),axis=0)

for x in x_patches_vec:
    x_strip_i = np.zeros([patch_size*upsample_factor,(patch_size-overlap)*len(x_patches_vec)*upsample_factor \
                          + overlap*upsample_factor], dtype=np.complex64)
    for y_i, y in enumerate(x_patches_vec):
        patch_i = all_high_res_guess_1[count]
        if y_i == 0: #first patch
            patch_i[:,-overlap*upsample_factor:] = patch_i[:,-overlap*upsample_factor:]*(1-indices)
            x_strip_i[:,0:patch_size*upsample_factor] = patch_i
        elif y_i == (len(x_patches_vec) - 1): #last patch
            patch_i[:,0:overlap*upsample_factor] = patch_i[:,0:overlap*upsample_factor]*indices
            x_strip_i[:,y_i*(patch_size-overlap)*upsample_factor:] = \
                x_strip_i[:,y_i*(patch_size-overlap)*upsample_factor:] + patch_i
        else: #middle patch
            patch_i[:,0:overlap*upsample_factor] = patch_i[:,0:overlap*upsample_factor]*indices
            patch_i[:,-overlap*upsample_factor:] = patch_i[:,-overlap*upsample_factor:]*(1-indices)
            
            x_strip_i[:,y_i*(patch_size-overlap)*upsample_factor:\
                      y_i*(patch_size-overlap)*upsample_factor+patch_size*upsample_factor] = \
                      x_strip_i[:,y_i*(patch_size-overlap)*upsample_factor:\
                      y_i*(patch_size-overlap)*upsample_factor+patch_size*upsample_factor] + \
                      patch_i
        count += 1
    x_strips.append(x_strip_i)


### merge together all rows
total_N_obj = (patch_size-overlap)*upsample_factor*len(x_patches_vec)+overlap*upsample_factor     
final_obj =  np.zeros([total_N_obj, total_N_obj],dtype=np.complex64)  
for y_i,y in enumerate(x_patches_vec):
    strip_i = x_strips[y_i]
    if y_i == 0: #first strip
        strip_i[-overlap*upsample_factor:,:] = strip_i[-overlap*upsample_factor:,:]*np.transpose((1-indices))
        final_obj[0:patch_size*upsample_factor,:] = strip_i
    elif y_i == (len(x_patches_vec) - 1): #last strip
        strip_i[0:overlap*upsample_factor,:] = strip_i[0:overlap*upsample_factor,:]*np.transpose(indices)
        final_obj[y_i*(patch_size-overlap)*upsample_factor:,:] = \
                final_obj[y_i*(patch_size-overlap)*upsample_factor:,:] + strip_i
    else: #middle strip
        strip_i[0:overlap*upsample_factor,:] = strip_i[0:overlap*upsample_factor,:]*np.transpose(indices)
        strip_i[-overlap*upsample_factor:,:] = strip_i[-overlap*upsample_factor:,:]*np.transpose((1-indices)) 
        final_obj[y_i*(patch_size-overlap)*upsample_factor:\
                      y_i*(patch_size-overlap)*upsample_factor+patch_size*upsample_factor,:] = \
                      final_obj[y_i*(patch_size-overlap)*upsample_factor:\
                      y_i*(patch_size-overlap)*upsample_factor+patch_size*upsample_factor,:] + \
                      strip_i

  
### Get low res images

low_res_stack_actual_ave = read_images(training_data_folder, total_N_obj/upsample_factor, [nstart_x_0,nstart_y_0], False, 0, num_stacks)

low_res_stack_actual = np.zeros([num_stacks, low_res_stack_actual_ave.shape[0], low_res_stack_actual_ave.shape[1], \
                                 low_res_stack_actual_ave.shape[2]], dtype=np.uint16)
for stack_i in range(1,num_stacks+1):
    low_res_stack_actual[stack_i-1,:,:,:] = read_images_single_stack(training_data_folder, total_N_obj/upsample_factor, [nstart_x_0,nstart_y_0], False, 0, stack_i)

### If folder contains "LEDPattern" then save those as well
num_LEDPattern_imgs = len(glob.glob(training_data_folder + "/LEDPattern/Photo000*.png"))

if num_LEDPattern_imgs>0:
    LEDPattern_stack = read_images_single_stack(training_data_folder, total_N_obj/upsample_factor, [nstart_x_0,nstart_y_0], False, 0, 0, LEDPattern = True)
    LEDPattern = True
else:
    LEDPattern = False
    LEDPattern_stack = None


### Make figures

CreateFig(np.sum(low_res_stack_actual_ave, axis=0),'low_res_stack_actual_sum',training_data_folder,title='low_res_stack_actual sum', vmin=50000, vmax=350000)
CreateFig(low_res_stack_actual_ave[34,:,:],'low_res_stack_actual_img0',training_data_folder,title='low_res_stack_actual img0')

#CreateFig(np.sum(low_res_stack_actual[0,:,:,:], axis=0),'low_res_stack_single_stack',training_data_folder,title='low_res_stack_single_stack', vmin=50000, vmax=350000)

CreateFig(np.abs(final_obj),\
          'final_obj_abs',training_data_folder,title='final_obj_abs')
CreateFig(np.angle(final_obj),\
          'final_obj_angle',training_data_folder,title='final_obj_angle')

### Save Data

numLEDs = low_res_stack_actual.shape[1]

#np.save(training_data_folder + '/low_res_stack_actual_ave.npy', low_res_stack_actual_ave)
np.save(training_data_folder + '/low_res_stack_actual.npy', low_res_stack_actual)
np.save(training_data_folder + '/final_obj.npy', final_obj)


#np.save(training_data_folder + '/Np.npy', total_N_obj/upsample_factor)
#np.save(training_data_folder + '/N_obj.npy', total_N_obj)
#np.save(training_data_folder + '/numLEDs.npy', numLEDs)


# Output low-resolution image
low_res_stack_actual_ave = np.round(low_res_stack_actual_ave)
low_res_stack_actual_ave[low_res_stack_actual_ave > (2**16 - 1)] = 2**16 - 1
low_res_stack_actual_ave[low_res_stack_actual_ave < 0] = 0
low_res_stack_actual_ave = low_res_stack_actual_ave.astype(np.uint16)


if 0:
    lowres_training_dataset = np.expand_dims(np.transpose(low_res_stack_actual_ave, axes=[1,2,0]),axis=0)
    training_dataset = np.expand_dims(final_obj,axis=0)
    
    lowres_training_dataset = lowres_training_dataset[:,0:464*2,0:464*2,:]
    training_dataset = training_dataset[:,0:928*2,0:928*2]
    
    np.save(training_data_folder + '/lowres_training_dataset.npy', \
            lowres_training_dataset)
    np.save(training_data_folder + '/training_dataset.npy', training_dataset)
    
    np.save(training_data_folder + '/Np.npy', 464*2)
    np.save(training_data_folder + '/N_obj.npy', 928*2)
    np.save(training_data_folder + '/numLEDs.npy', numLEDs)



N_obj_save = total_N_obj/save_patches
Np_save = total_N_obj/upsample_factor/save_patches
np.save(training_data_folder +  '/Np.npy', Np_save)
np.save(training_data_folder +  '/N_obj.npy', N_obj_save)
np.save(training_data_folder +  '/numLEDs.npy', numLEDs)

lower_bnd = -50
upper_bnd = 50
bnds =np.array([np.max(np.real(final_obj)), np.min(np.real(final_obj)), np.max(np.imag(final_obj)),np.min(np.imag(final_obj))])

if np.min(bnds) < lower_bnd:
    print('Warning, truncating final_obj, lower bound.')
if np.max(bnds) > upper_bnd:
    print('Warning, truncating final_obj, upper bound.')


patch_num = 0
for p_x in range(save_patches):
    for p_y in range(save_patches):
        save_pngs_patch(p_x, p_y, patch_num, Np_save, N_obj_save, numLEDs, \
                    training_data_folder, low_res_stack_actual_ave, final_obj, \
                    lower_bnd, upper_bnd, LEDPattern, LEDPattern_stack)
        patch_num += 1







### reconstruct complex obj from pngs
#real_part = imageio.imread(highres_real_filename)
#imag_part = imageio.imread(highres_imag_filename)
#
#real_part = unprocess_final_obj(real_part)
#imag_part = unprocess_final_obj(imag_part)
#
#img = real_part + 1j*imag_part
#plt.figure()
#plt.imshow(np.abs(img))
#
#plt.figure()
#plt.imshow(np.angle(img))