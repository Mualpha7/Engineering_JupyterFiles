#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 18:23:52 2018

@author: ycheng2 

FP_Low_Res_Reconstruction combines FPSystemSetup.py and FP_Optimizer_lowResInput.py into one file 
Purpose: Allow for tuning of parameters using Tensorflow 
"""

# Import necessary functions 


import TensorFlowFunctions as tff 
from layer_defs import variable_on_cpu
from FP_Reconstruction_Functions_3D import read_images, derived_params, \
                                        create_low_res_stack_singleLEDs_multipupil_3D, \
                                        CalculateParameters, NAfilter, scalar_prop
from ShiftAddFunctions import get_derived_params_SA, shift_add

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from scipy import signal
import os, argparse, time, sys

version = 57
print(version)

### Accept command line inputs 
##############################################################################

# Function to parse input from command line 
# Inputs 
# --data: Path for input unprocessed dataset
# --folder: Path for output reconstruction images 
# --size-leds: Diameter of illumination 
def parseCommandLineInput():
    parser = argparse.ArgumentParser(description='Get command line args')
    
    parser.add_argument('--input', action='store', help='path for input unprocessed dataset (images)')
    
    parser.add_argument('--output', action='store', help='path for output dataset')
    
    parser.add_argument('-g', type=int, action='store', dest='num_gpus', \
                        help='number of GPUs')
    
    parser.add_argument('--np', type=int, action='store', dest='np', \
                        help='Np = side length of low resolution image ')
    
    parser.add_argument('--nstart_x', type=int, action='store', dest='nstart_x', \
                        help='Starting coordinate for defining region of interest, row coordinate')    

    parser.add_argument('--nstart_y', type=int, action='store', dest='nstart_y', \
                        help='Starting coordinate for defining region of interest, column coordinate')    

    parser.add_argument('--background', type=float, action='store', dest='background_threshold', \
                            help='background_threshold')
        
    return parser 

# Parse input from command line 
parser = parseCommandLineInput()
# Store input arguments 
args = parser.parse_args()

# Define input folder 
input_folder_name = args.input

# Define output folder
output_folder = args.output 
        
# Side length of low resolution image 
Np = args.np

 # Starting coordinate for defining region of interest
nstart = [args.nstart_x, args.nstart_y]

background_threshold = args.background_threshold

# Number of GPUs to use for training   
num_GPUs = args.num_gpus 


### Define processing Region of Interest 
##############################################################################
upsample_factor = 2 # XXX add a check that makes sure this factor is large enough
                    # upsampling factor must support NA = 2
                    # 6
N_obj = Np*upsample_factor


# Define whether to perform background removal 
#############################################################################
background_removal = True


### Define parameters describing the optical system 
##############################################################################
# Wavelength of illumination 
# Adjust accordingly 
wavelength = 0.518 
# R: 624.4nm +- 50nm 
# G: 518.0nm +- 50nm 
# B: 476.4nm +- 50nm 

# Numerical Aperture of Objective
# Adjust accordingly 
NA = 0.3 #0.5 for 20x and 0.3 for 10x 
print("Initial system NA is ", NA)

# Magnification of the system 
# Adjust accodingly 
mag = 9.24 #9.24 for 10x #18.48 for 20x

# Diameter of pixel on sensor plane 
# Adjust accordingly
dpix_c = 6.5 # 6.5um pixel size on the sensor plane 

# LED array geometries
# Adjust accordingly 
ds_led = 4e3 # 4mm, spacing between neighboring LEDs
z_led = 69.5e3 # Z distance between LED and imaged object 
lit_cenv = 15 # center LED in the row direction
lit_cenh = 16 # center LED in the column direction
dia_led = 9.0 # diameter of used LEDs 


N_obj_center = [2048*upsample_factor//2, 2048*upsample_factor//2] #coords of object center, corresponds to center LED


# Define patch parameters 
num_patches = 1 
N_patch = N_obj//num_patches #patches that make up the high res object


### Define training parameters 
# Adjust accordingly 
##############################################################################
training_rate = 2e-1 # Rate of training 
sqrt_reg = 1e-8 # Square-root regularization 
num_iter = 10 # Number of iterations to train for 
restore_model = False # Whether to restore the model for every restore_iter
restore_iter = 20000 # Graph restoration point 
save_checkpoint_interval = 100000 # Checkpoint saving interval defined
save_vars_interval = 1000 #100000 # Variable saving interval defined 
change_pupil = True
change_Ns_mat = False ### This option doesn't work for 3D, don't change from False
change_scale_mat = False
num_stacks = 1

# Get collected low-resolution images

Imea = read_images(input_folder_name, Np, nstart, background_removal, background_threshold, num_stacks)

# Get derived parameters

numLEDs,dx_obj, hhled, vvled, du, LitCoord, um_m, pupil = derived_params(NA, wavelength, dpix_c, mag, Np, \
                                                                         lit_cenh, lit_cenv, dia_led, N_obj)

# Obtain the values for Ns_mat and scale_mat 

Ns_mat = np.zeros([num_patches**2,numLEDs,2])
scale_mat = np.zeros([num_patches**2,numLEDs])
count = 0
for i,startX in enumerate(np.arange(0,N_obj,N_patch)):
    for j,startY in enumerate(np.arange(0,N_obj,N_patch)):
        N_obj_patch_coord = np.array([startX,startY]) + np.array(nstart)*upsample_factor # upper left corner of object patch which will be downconverted to low-resolution
        N_patch_center = N_obj_patch_coord + N_obj/num_patches/2 
        
        # pass the full object to HiToLoPatch
        Ns, scale, synthetic_NA = CalculateParameters(N_patch_center, N_obj_center, dx_obj, hhled, ds_led, \
                        vvled, z_led, wavelength, du, LitCoord, numLEDs, NA, um_m)
        
        Ns_mat[count,:,:]=Ns
        scale_mat[count,:]=scale
        count += 1
        
        print('synthetic_NA: ', synthetic_NA)


# Determine synthetic NAfilter
NAfilter_synthetic = NAfilter(int(N_obj),dx_obj*N_obj*1e-6,wavelength*1e-6,synthetic_NA)

# Determine NAfilter for NA=2
NAfilter_2 = NAfilter(int(N_obj),dx_obj*N_obj*1e-6,wavelength*1e-6,2.0)


# Determine NAfilter for NA=1 + illumination NA
NAfilter_3 = NAfilter(int(N_obj),dx_obj*N_obj*1e-6,wavelength*1e-6,1+synthetic_NA-NA)

# Display Pre-processing data and synthetic NA 
##############################################################################


## print synthetic NA
#plt.figure()
#plt.imshow(NAfilter_synthetic)
#plt.colorbar()
#
## print NA = 2, if it doesn't fit, increase the upsample factor
#plt.figure()
#plt.imshow(NAfilter_2)
#plt.colorbar()
#
## print NAfilter_3
#plt.figure()
#plt.imshow(NAfilter_3)
#plt.colorbar()
#
#
## Show low-resolution images
#plt.figure()
#plt.imshow(Imea[5])
#plt.figure()
#plt.imshow(Imea[10])



### Pre-process variables for reconstruction
# Slight reformatting of variables  
##############################################################################
# cen0: Coordinates for object patch center
cen0 = np.array([N_obj//2,N_obj//2], dtype=np.float32) #N_obj_center


# Np: Lower resolution side length 
Np = int(Np)

### Define a training folder
##############################################################################
### Make output folder
training_folder = input_folder_name + '/' + output_folder
try: 
    os.makedirs(training_folder)
except OSError:
    if not os.path.isdir(training_folder):
        raise




# Save Inputs
#############################################################################
np.savez(training_folder + '/input_vars.npz',                        
         Np = Np,
         nstart = nstart,
         background_threshold = background_threshold,
         num_GPUs = num_GPUs)


# Make GPU_devices_vec 
#############################################################################
if num_GPUs: 
    GPU_devices_vec=[];
    for num in list(range(num_GPUs)) :
        GPU_devices_vec.append('/device:GPU:'+str(num))
else:
    GPU_devices_vec=['/cpu:0']
    num_GPUs = 1 



indices0=[]
for g in list(range(num_GPUs)):
    if g == (num_GPUs-1):
        indices0.append(list(range(g*(numLEDs//num_GPUs),numLEDs)))
    else:    
        indices0.append(list(range(g*(numLEDs//num_GPUs),(g+1)*(numLEDs//num_GPUs))))
 
# Load z variables

dz = np.load(input_folder_name + '/dz.npy')
z_vec = np.load(input_folder_name + '/z_vec.npy')
Nz = len(z_vec)

# Initial condition

tot_mat = np.load(input_folder_name + '/tot_mat.npy')
tot_mat = tot_mat[:,nstart[0]:nstart[0]+Np,nstart[1]:nstart[1]+Np]

bits = 2**16 - 1
       
initial_guess = np.sqrt(tot_mat/float(numLEDs))
initial_guess = initial_guess/float(upsample_factor**2) 
initial_guess = initial_guess**(1./float(Nz))    
#initial_guess = initial_guess/100.0   
initial_guess = signal.resample(initial_guess, N_obj, axis=1)
initial_guess = signal.resample(initial_guess, N_obj, axis=2)                      
initial_guess = initial_guess.astype(np.float32)

# Create H_prop
H_prop_dz_np = scalar_prop(N_obj,N_obj,dx_obj,dx_obj,dz,wavelength)
H_prop_focal_plane_np = scalar_prop(N_obj,N_obj,dx_obj,dx_obj,-z_vec[-1],wavelength)
      
np.save(training_folder + '/low_res_stack_actual.npy', Imea)
np.save(training_folder + '/LitCoord.npy', LitCoord)

#sys.exit()

# Start Timer
trainingStart=time.time()

# Tensorflow Graph
with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.device('/cpu:0'):
        
        H_prop_dz = tf.constant(H_prop_dz_np, dtype=tf.complex64)
        
        H_prop_focal_plane = tf.constant(H_prop_focal_plane_np, dtype=tf.complex64)
        
        # actual low-resolution images
        low_res_stack_actual = tf.constant(Imea[0:numLEDs,:,:], dtype=tf.float32)
    
        
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    
        # Decay the learning rate exponentially based on the number of steps.
        training_rate_decay = tf.train.exponential_decay(training_rate,
                                        global_step,
                                        1000, # decay_steps 
                                        0.999, # decay_rate
                                        staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=training_rate_decay, epsilon=1e-3) #1e-2 #1e-3 #1e-4
    
    
    ##############
        tower_grads = []
        sum_tower_loss = []
        low_res_stack_predicted_dict = {}


        with tf.variable_scope(tf.get_variable_scope()):
            print(tf.get_variable_scope())
            for g,gpu in enumerate(GPU_devices_vec):
                with tf.device(gpu):
                    
                    if change_pupil:
                        
                        pupil0 = tf.expand_dims(tf.constant(pupil, dtype = tf.complex64),axis=0)
                        
                        
                        P_angle = variable_on_cpu('pupil_angle', \
                                                  np.zeros([num_patches**2,int(Np),int(Np)]).astype(np.float32), \
                                                  tf.float32)
                        
                        P = tf.exp(tf.cast(P_angle, tf.complex64)*1j)*pupil0
                    else:
                        
                        P = tf.constant(pupil, dtype = tf.complex64)
    
                    if change_Ns_mat:
                        Ns_mat_tf = variable_on_cpu('Ns_mat_tf', \
                                                  Ns_mat.astype(np.float32), \
                                                  tf.float32)
                    else:
                        Ns_mat_tf = tf.constant(Ns_mat, dtype=tf.float32)
                    
                    
                    if change_scale_mat:
                        scale_mat_tf = variable_on_cpu('scale_mat_tf', \
                                                  scale_mat.astype(np.float32), \
                                                  tf.float32)
                    else:
                        scale_mat_tf = tf.constant(scale_mat, dtype=tf.float32)
                


                    high_res_guess_real = variable_on_cpu('high_res_guess_real', \
                                                          initial_guess,
                                                          tf.float32)
                    high_res_guess_img = variable_on_cpu('high_res_guess_img', \
                                                         np.zeros([int(Nz),int(N_obj),int(N_obj)], dtype=np.float32), 
                                                         tf.float32)
                    
                    
                    
                    high_res_guess = tf.cast(high_res_guess_real, tf.complex64) + 1j*tf.cast(high_res_guess_img, tf.complex64)
                    
                
                    low_res_stack_predicted = create_low_res_stack_singleLEDs_multipupil_3D(high_res_guess, H_prop_dz, H_prop_focal_plane, \
                                                                                            Nz, numLEDs, N_obj, \
                                                                                            N_patch, num_patches, Ns_mat_tf, scale_mat_tf, \
                                                                                            cen0, P, Np, indices0[g], change_pupil, change_Ns_mat)
                    
                    # compare the stack of low resolution images to the actual low resolution images
                   
                    low_res_stack_actual0 = tf.gather(low_res_stack_actual, indices0[g])

#                    loss_MSE = tf.reduce_sum(tf.square(tf.sqrt(low_res_stack_actual0 + sqrt_reg) - tf.sqrt(low_res_stack_predicted + sqrt_reg)))
                    loss_MSE = tf.reduce_sum(tf.square(tf.sqrt(low_res_stack_actual0) - tf.sqrt(low_res_stack_predicted)))
#                    loss_MSE = tf.reduce_sum(tf.square(low_res_stack_actual0 - low_res_stack_predicted))

#                    loss_grad_diff = tff.grad_diff_loss(low_res_stack_predicted, low_res_stack_actual0)
#                    
#                    if change_pupil:
#                        alpha = 1e-1
#                        pupil_change_penalty = alpha*tf.reduce_sum(tf.square(tf.abs(pupil0-P)))
#                        loss_i = loss_MSE + loss_grad_diff*0 + pupil_change_penalty*0
#                    else:
#                        loss_i = loss_MSE + loss_grad_diff*0
                
                    loss_i = loss_MSE
                    
                    loss_i = loss_i*len(indices0[g])*num_GPUs/float(numLEDs)
                    
                    grads = optimizer.compute_gradients(loss_i)
                    
                    tower_grads.append(grads)
                    sum_tower_loss.append(loss_i)
                    low_res_stack_predicted_dict[g] = low_res_stack_predicted
                    
                    tf.get_variable_scope().reuse_variables()


        loss=tf.add_n(sum_tower_loss)
        
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = tff.average_gradients(tower_grads, take_average=False)

    
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
            
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()  
  
        config = tf.ConfigProto(
                device_count = {'GPU': num_GPUs},
                allow_soft_placement=True,
                log_device_placement=False
                )    
  
        
    with tf.Session(config=config) as sess: 
        if restore_model:
            
            saver.restore(sess, training_folder + '/model.ckpt-' + str(restore_iter))
            print("Model restored.")
            
            iter_vec = np.load(training_folder + '/iter_vec.npy').tolist()
            
            
        else:
            restore_iter = -1
            
            sess.run(init_op)
            
            iter_vec=[]


        for i in list(range(restore_iter + 1, restore_iter + 1 + num_iter)):
                    
            if ( ((i%save_vars_interval)==0) or (i==(restore_iter + 1 + num_iter - 1))):

                folder_name = training_folder
                    

                [ _ ,
                 loss0,
                 high_res_guess0,
                 P0,
                 Ns_mat0,
                 scale_mat0,
                 low_res_stack_predicted_dict0]= sess.run([apply_gradient_op,
                                                           loss,
                                                           high_res_guess,
                                                           P,
                                                           Ns_mat_tf,
                                                           scale_mat_tf,
                                                           low_res_stack_predicted_dict])
                  
                                     
                np.save(folder_name + '/loss' + str(i) + '.npy', loss0)
                np.save(folder_name + '/high_res_guess' + str(i) + '.npy', high_res_guess0)
                np.save(folder_name + '/P' + str(i) + '.npy', P0)
                np.save(folder_name + '/Ns_mat' + str(i) + '.npy', Ns_mat0)
                np.save(folder_name + '/scale_mat' + str(i) + '.npy', scale_mat0)
                np.save(folder_name + '/low_res_stack_predicted_dict' + str(i) + '.npy', low_res_stack_predicted_dict0)

                
                iter_vec.append(i) 
                np.save(folder_name + '/iter_vec.npy', iter_vec)
                
                                      
                print('iteration: ',i) 
                print('loss training:   ', loss0)            



            else:
                sess.run(apply_gradient_op) 

 
            if ( ((i%save_checkpoint_interval)==0) and (i>0) ): # or (i==(restore_iter + 1 + num_iter - 1))):
                save_path = saver.save(sess, training_folder + '/model.ckpt',global_step=i)
                print("Model saved in file: %s" % save_path)  
          
### End Timer    
trainingEnd=time.time()
trainingTotalTime=trainingEnd-trainingStart
print('Training took', trainingTotalTime, 'seconds.')
np.save(folder_name + '/training_time.npy', trainingTotalTime)

print(version)