#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:55:20 2018

@author: vganapa1
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def load_final_var(var_name, folder_name, iter_vec):
    var_1 = np.load(folder_name + '/' + var_name + str(iter_vec[-1]) + '.npy')
    return var_1

def CreateFig(mat,save_name,folder_name,title='', vmin=None, vmax=None): #folder_name is where the figure is saved
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(mat, interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(folder_name+'/'+save_name+'.png',bbox_inches='tight', dpi = 300) 


def process_final_obj(final_obj_part, lower_bnd, upper_bnd): # can input real or imag part
    final_obj_part = (final_obj_part - lower_bnd)*(2**16-1.)/float(upper_bnd - lower_bnd)
    # truncate below 0
    final_obj_part[np.nonzero(final_obj_part < 0)] = 0
    # truncate above 2**16-1
    final_obj_part[np.nonzero(final_obj_part > (2**16-1))] = 2**16-1
    final_obj_part = final_obj_part.astype(np.uint16)
    return final_obj_part
    
def unprocess_final_obj(final_obj_part, upper_bnd, lower_bnd):
    return final_obj_part*float(upper_bnd - lower_bnd)/(2**16-1.) + lower_bnd


def save_pngs_patch(p_x, p_y, patch_num, Np_save, N_obj_save, numLEDs, \
                    training_data_folder, low_res_stack_actual_ave, final_obj, \
                    lower_bnd, upper_bnd, LEDPattern, LEDPattern_stack):
    
    save_folder = training_data_folder + '/' + training_data_folder + '_patch_' + str(patch_num)
    
    try: 
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise
    
    #Output low_resolution images
    
    for LED_i in range(numLEDs):
        lowres_filename = save_folder + '/lowres' + str(LED_i)  + '.png'
        imageio.imwrite(lowres_filename, low_res_stack_actual_ave[LED_i,p_x*Np_save:(p_x+1)*Np_save,p_y*Np_save:(p_y+1)*Np_save])
    
    #Output LEDPattern images
    
    if LEDPattern:
        for num_img in range(LEDPattern_stack.shape[0]):
            lowres_opt_filename = save_folder + '/lowres_opt' + str(num_img)  + '.png'
            imageio.imwrite(lowres_opt_filename, LEDPattern_stack[num_img,p_x*Np_save:(p_x+1)*Np_save,p_y*Np_save:(p_y+1)*Np_save])
            
    # Output high-resolution object

    final_obj_real = process_final_obj(np.real(final_obj[p_x*N_obj_save:(p_x+1)*N_obj_save,p_y*N_obj_save:(p_y+1)*N_obj_save]), lower_bnd, upper_bnd)
    
    final_obj_imag = process_final_obj(np.imag(final_obj[p_x*N_obj_save:(p_x+1)*N_obj_save,p_y*N_obj_save:(p_y+1)*N_obj_save]), lower_bnd, upper_bnd)
    
    highres_real_filename = save_folder +  '/highres_real.png'
    highres_imag_filename = save_folder +  '/highres_imag.png'
    
    imageio.imwrite(highres_real_filename,final_obj_real)
    imageio.imwrite(highres_imag_filename,final_obj_imag)
    
    return highres_real_filename, highres_imag_filename

def save_pngs_patch_2(p_x, p_y, patch_num, Np_save, N_obj_save, numLEDs, \
                    training_data_folder, low_res_stack_actual_ave, final_obj, \
                    lower_bnd, upper_bnd, LEDPattern, LEDPattern_stack):
    
    save_folder = training_data_folder + '/' + training_data_folder + '_rc_patch_' + str(patch_num)
    
    try: 
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise
    
    #Output low_resolution images
    
    for LED_i in range(numLEDs):
        lowres_filename = save_folder + '/lowres' + str(LED_i)  + '.png'
        imageio.imwrite(lowres_filename, low_res_stack_actual_ave[LED_i,p_x*Np_save/2:p_x*Np_save/2+Np_save,p_y*Np_save/2:p_y*Np_save/2+Np_save])
    
    #Output LEDPattern images
    
    if LEDPattern:
        for num_img in range(LEDPattern_stack.shape[0]):
            lowres_opt_filename = save_folder + '/lowres_opt' + str(num_img)  + '.png'
            imageio.imwrite(lowres_opt_filename, LEDPattern_stack[num_img,p_x*Np_save/2:p_x*Np_save/2+Np_save,p_y*Np_save/2:p_y*Np_save/2+Np_save])
            
    # Output high-resolution object

    final_obj_real = process_final_obj(np.real(final_obj[p_x*N_obj_save/2:p_x*N_obj_save/2+N_obj_save,p_y*N_obj_save/2:p_y*N_obj_save/2+N_obj_save]), lower_bnd, upper_bnd)
    
    final_obj_imag = process_final_obj(np.imag(final_obj[p_x*N_obj_save/2:p_x*N_obj_save/2+N_obj_save,p_y*N_obj_save/2:p_y*N_obj_save/2+N_obj_save]), lower_bnd, upper_bnd)
    
    highres_real_filename = save_folder +  '/highres_real.png'
    highres_imag_filename = save_folder +  '/highres_imag.png'
    
    imageio.imwrite(highres_real_filename,final_obj_real)
    imageio.imwrite(highres_imag_filename,final_obj_imag)
    
    return highres_real_filename, highres_imag_filename