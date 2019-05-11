#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:31:49 2018

@author: vganapa1
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

### Command Line Inputs

parser = argparse.ArgumentParser(description='Get command line args')

### Inputs

parser.add_argument('-d', type=str, action='store', dest='dataset_folder', \
                        help='folder containing training, validation, and test images', \
                        default = 'Flatworm_SampleNumber0002_RegionNumber0001') #_SampleNumber0002_RegionNumber0001

parser.add_argument('-o', type=str, action='store', dest='training_output_folder', \
                        help='folder containing output from training/validation/test', \
                        default = 'patch_4')



args = parser.parse_args()


### Inputs from command line args
training_data_folder = args.dataset_folder 
folder_name = training_data_folder + '/' + args.training_output_folder

### Other inputs
z_ind=2
num_GPUs = 4


### Functions

def CreateFig(mat,save_name,folder_name,title='', vmin=None, vmax=None): #folder_name is where the figure is saved
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(mat, interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(folder_name+'/'+save_name+'.png',bbox_inches='tight', dpi = 300) 
    plt.show()


### Load variables

#common_strip_1 = high_res_guess_p1[:,-64*2:]
#common_strip_2 = high_res_guess_p2[:,0:64*2]
#indices = np.arange(0,128)/128.
#common_strip = common_strip_1*(1-indices)+common_strip_2*(indices)
#a=np.concatenate((high_res_guess_p1[:,0:-128],common_strip,high_res_guess_p2[:,128:]),axis=1)
#CreateFig(np.abs(a),'high_res_stitched',folder_name,title='high_res_stitched')

#a=np.concatenate((high_res_guess_p1,high_res_guess_p2),axis=1)
#CreateFig(np.abs(a),'high_res_stitched',folder_name,title='high_res_stitched')

input_vars = np.load(folder_name + '/input_vars.npz')
iter_vec = np.load(folder_name + '/iter_vec.npy')

tot_mat = np.load(training_data_folder + '/tot_mat.npy')

initial_iter = 0
final_iter = -1

def load_vars(var_name):
    var_0 = np.load(folder_name + '/' + var_name + str(iter_vec[initial_iter]) + '.npy')
    var_1 = np.load(folder_name + '/' + var_name + str(iter_vec[final_iter]) + '.npy')
    return var_0, var_1

def combine_low_res_stack(low_res_stack_predicted_dict):
    low_res_stack_predicted = []
    for g in list(range(num_GPUs)):
        low_res_stack_predicted.append(low_res_stack_predicted_dict[g])
        
    return np.concatenate(low_res_stack_predicted, axis=0)  
    

loss_0, loss_1 = load_vars('loss')

high_res_guess_0, high_res_guess_1 = load_vars('high_res_guess')

P_0, P_1 = load_vars('P')

scale_mat_0, scale_mat_1 = load_vars('scale_mat')

Ns_mat_0, Ns_mat_1 = load_vars('Ns_mat')

low_res_stack_predicted_dict_0, low_res_stack_predicted_dict_1 = load_vars('low_res_stack_predicted_dict')
low_res_stack_predicted_dict_0 = low_res_stack_predicted_dict_0.flat[0]
low_res_stack_predicted_dict_1 = low_res_stack_predicted_dict_1.flat[0]

#low_res_stack_predicted_0 = combine_low_res_stack(low_res_stack_predicted_dict_0)
#low_res_stack_predicted_1 = combine_low_res_stack(low_res_stack_predicted_dict_1)
 
low_res_stack_predicted_0 = low_res_stack_predicted_dict_0
low_res_stack_predicted_1 = low_res_stack_predicted_dict_1

### Get low res images

low_res_stack_actual = np.load(folder_name + '/low_res_stack_actual.npy')

### Create Figures

CreateFig(np.sum(low_res_stack_actual, axis=0),'low_res_stack_actual_sum',folder_name,title='low_res_stack_actual sum')
CreateFig(low_res_stack_actual[34,:,:],'low_res_stack_actual_img0',folder_name,title='low_res_stack_actual img0')

CreateFig(np.sum(low_res_stack_predicted_0[0], axis=0),'low_res_stack_predicted_0_sum',folder_name,title='low_res_stack_predicted_0 sum')
CreateFig(low_res_stack_predicted_0[0][34,:,:],'low_res_stack_predicted_0_img0',folder_name,title='low_res_stack_predicted_0 img0')

CreateFig(np.sum(low_res_stack_predicted_1[0], axis=0),'low_res_stack_predicted_1_sum',folder_name,title='low_res_stack_predicted_1 sum')
CreateFig(low_res_stack_predicted_1[0][34,:,:],'low_res_stack_predicted_1_img0',folder_name,title='low_res_stack_predicted_1 img0')


CreateFig(np.abs(high_res_guess_0[z_ind,:,:]),'high_res_guess_0_abs',folder_name,title='high_res_guess_0_abs')
CreateFig(np.angle(high_res_guess_0[z_ind,:,:]),'high_res_guess_0_angle',folder_name,title='high_res_guess_0_angle')


#for z in range(5):
#    z_ind=z
CreateFig(np.abs(high_res_guess_1[z_ind,:,:]),\
          'high_res_guess_1_abs',folder_name,title='high_res_guess_1_abs')
CreateFig(np.angle(high_res_guess_1[z_ind,:,:]),\
          'high_res_guess_1_angle',folder_name,title='high_res_guess_1_angle')

if len(P_0.shape)==3:
    CreateFig(np.abs(P_0[0,:,:]),\
              'P_0_abs',folder_name,title='P_0_abs')
    CreateFig(np.angle(P_0[0,:,:]),\
              'P_0_angle',folder_name,title='P_0_angle')
        
        
    CreateFig(np.abs(P_1[0,:,:]),\
              'P_1_abs',folder_name,title='P_1_abs')
    CreateFig(np.angle(P_1[0,:,:]),\
              'P_1_angle',folder_name,title='P_1_angle')



### Graph of loss
loss_vec=[]

for iter_num in iter_vec:
    loss_i=np.load(folder_name + '/' + 'loss' + str(iter_num) + '.npy')   
    loss_vec.append(loss_i)

plt.figure()
plt.title('loss')
plt.plot(iter_vec,np.log(loss_vec))
plt.savefig(folder_name+'/loss.png',bbox_inches='tight')
plt.show() 
