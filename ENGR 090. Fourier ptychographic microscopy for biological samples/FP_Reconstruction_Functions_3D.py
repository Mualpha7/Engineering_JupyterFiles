#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:42:45 2018

@author: vganapa1
"""

import imageio
import glob
import numpy as np
import tensorflow as tf 
import math

def fftshift(mat2D, dim0, dim1): #fftshift == ifftshift when dimensions are all even
                                 #fftshift only works with even dimensions

    if (dim0==1) and (dim1==1):
        return mat2D    
    
    if (dim0%2) or (dim1%2):
        raise ValueError('Dimensions must be even to use fftshift.')

    dim0=tf.cast(dim0,tf.int32)
    dim1=tf.cast(dim1,tf.int32)

    piece1=tf.slice(mat2D,[0,0],[dim0//2,dim1//2])
    piece2=tf.slice(mat2D,[0,dim1//2],[dim0//2,dim1//2])
    piece3=tf.slice(mat2D,[dim0//2,0],[dim0//2,dim1//2])
    piece4=tf.slice(mat2D,[dim0//2,dim1//2],[dim0//2,dim1//2])

    top=tf.concat([piece4,piece3],axis=1)
    bottom=tf.concat([piece2,piece1],axis=1)

    final=tf.concat([top,bottom],axis=0)
    return final

def F(mat2D,dim0,dim1): # Fourier transform, origin in center
    return fftshift(tf.fft2d(fftshift(mat2D,dim0,dim1)),dim0,dim1)

def Ft(mat2D,dim0,dim1): # inverse Fourier transform, origin in center
    return fftshift(tf.ifft2d(fftshift(mat2D,dim0,dim1)),dim0,dim1)


def scalar_prop(Nx,Ny,dx,dy,dz,wavelength): # x is the row coordinate, y is the column coordinate

    Lx = Nx*dx
    Ly = Ny*dy
    
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Nx) #freq coords
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,Ny) #freq coords
    
    FX,FY=np.meshgrid(fx,fy, indexing = 'ij')

    H_scalar_prop = np.exp(1j*2*math.pi*(1./wavelength)*dz*np.sqrt(1-(wavelength*FX)**2+(wavelength*FY)**2))
    H_scalar_prop[np.nonzero( np.sqrt(FX**2+FY**2) > (1./wavelength) )] = 0

    return H_scalar_prop

def read_images(input_folder_name, Np, nstart, background_removal, threshold, num_stacks):
    #returns average over all stacks
    
    for stack_i in range(1,num_stacks+1):
        if stack_i == 1:
            Imea_ave = read_images_single_stack(input_folder_name, Np, nstart, background_removal, threshold, stack_i)/float(num_stacks)
        else:
            Imea_ave = Imea_ave + read_images_single_stack(input_folder_name, Np, nstart, background_removal, threshold, stack_i)/float(num_stacks)
    
    
    return Imea_ave
    
def read_images_single_stack(input_folder_name, Np, nstart, background_removal, threshold, stack_i, LEDPattern = False):    
    ### Read in images into memory
    ##############################################################################
    # Raw image size 
    n1 = 2048
    n2 = 2048
    print("Loading in images...")
    
    if LEDPattern:
        # Get file names 
        filenames = glob.glob(input_folder_name + "/LEDPattern/Photo000*.png")
        
        # Get number of files 
        numImages = len(filenames)
    
        # Order file names 
        for i in range(0,numImages): 
            filenames[i] = input_folder_name + "/LEDPattern/Photo000" + str(i) + ".png"
            
    else:
        # Get file names 
        filenames = glob.glob(input_folder_name + "/Images/Stack000" + str(stack_i) + "/Photo*.png")
        
        # Get number of files 
        numImages = len(filenames)
    
        # Order file names 
        for i in range(0,numImages): 
            filenames[i] = input_folder_name + "/Images/Stack000" + str(stack_i) + "/Photo" + '0'*(4-len(str(i))) + str(i) + ".png"
            
        
    # Initialize list of images 
    Iall = np.zeros((numImages, n1, n2), dtype="uint16")
    # Initialize background information 
    Ibk = np.zeros((numImages, 1))
    
    # Iterate through all images and save images to list  
    for i in range(0, numImages):      
        img = imageio.imread(filenames[i])
        
        Iall[i,:,:] = img
        # Specify background region coordinates 
        bk1 = np.mean(Iall[i,200:250,200:250])
        bk2 = np.mean(Iall[i,1024:1124,1024:1124])
        Ibk[i] = np.mean((bk1, bk2))
        # Define background noise threshold  
        if Ibk[i] > threshold:
            if i > 0: 
                Ibk[i] = Ibk[i-1]
            else: 
                Ibk[i] = threshold
    print("Finish loading images")

    # Crop images according to region of interest 
    ##############################################################################
    # Crop images according to region of interest 
    Imea = Iall[:, nstart[0]:nstart[0]+Np, nstart[1]:nstart[1]+Np]
    # Cast image in preperation for processing 
    Imea.astype(float) 
    
    # Perform background noise removal
    ############################################################################## 
    if background_removal: 
        for i in range(0, numImages):
            Itmp = Imea[i,:,:] - Ibk[i]
            Itmp = np.maximum(Itmp, 0)
            Imea[i,:,:] = Itmp
    
    return Imea


def derived_params(NA,wavelength,dpix_c,mag,Np, lit_cenh, lit_cenv, dia_led, N_obj):
    ### Calculate derived parameters describing the optical system 
    ##############################################################################
    # Maximum spatial frequency of low-resolution images set by NA 
    um_m = NA/wavelength 
    # System resolution based on NA 
#    dx0 = 1/um_m/2 
    
    # Effective image pixel size on the image plane 
    dpix_m = dpix_c/mag 
    
    # FoV of object space
    # Field of view 1000
    # Low resolution object length multiplied by effective pixel size  
    FoV = Np*dpix_m
    
    # Sampling size 
    # Define sampling size at Fourier plane set by the image size (FoV)
    # Sampling size at the Fourier plane is 1/FoV 
    du = 1./FoV 
    
    # Low pass filter set-up 
    m = np.arange(0, Np, 1)
    # Generate a meshgrid 
    # mm: vertical
    # nn: horizontal 
    [mm,nn] = np.meshgrid(m-Np/2, m-Np/2)
    # Find radius of each pixel from center 
    ridx = np.sqrt(mm**2+nn**2)
    # The circle: max spatial frequency / sampling size in Fourier plane
    um_idx = um_m/du 
    # assume a circular pupil function, low pass filter due to finite NA
    w_NA = np.zeros(ridx.shape)
    w_NA[np.nonzero(ridx<um_idx)] = 1.
    # Define aberration 
    aberration = np.ones([int(Np), int(Np)])
    # Define phase constant 
    phC = np.ones([int(Np), int(Np)])
    # Generate pupil function 
    pupil = w_NA * phC * aberration 
    
    
    # LED matrix setup
    # Define LED coordinates 
    # h: horizontal (row), v: vertical (column)
    vled = np.array(range(0,32))-lit_cenv #32
    hled = np.array(range(0,32))-lit_cenh #32
    hhled,vvled = np.meshgrid(hled,vled)
    rrled = np.sqrt(hhled**2+vvled**2) # [LEDs] physical distance from center LED
    # Define coordinates in LED matrix that are actually used for imaging 
    LitCoord = np.zeros(hhled.shape)
    LitCoord[np.nonzero( rrled < (dia_led/2.) )] = 1. # LED array, with 1's for the LEDs actually used
#    Litidx = np.nonzero(LitCoord) # index of LEDs used in the experiment
    numLEDs = len(np.nonzero(LitCoord)[0]) # Total number of LEDs used in experiment
    
    dx_obj = FoV/N_obj
    
    return numLEDs, dx_obj, hhled, vvled, du, LitCoord, um_m, pupil


def create_low_res_stack_singleLEDs_multipupil_3D(high_res_obj_stack, H_prop_dz, H_prop_focal_plane, \
                                                  num_z_layers, numLEDs, N_obj, \
                                                  N_patch, num_patches, Ns_mat, scale_mat, \
                                                  cen0, P, Np, LED_vec, change_pupil, change_Ns_mat):
    
    scale_multiply = np.ones([numLEDs,])
    
    for LED_i in LED_vec: 
        low_res = HiToLo_singleLED_multipupil_3D(high_res_obj_stack, H_prop_dz, H_prop_focal_plane, num_z_layers,\
                                                 N_obj, N_patch, scale_multiply, num_patches,\
                                                 Ns_mat, scale_mat, cen0, P, Np, LED_i, \
                                                 change_pupil, change_Ns_mat)    
        
        low_res = tf.expand_dims(low_res, axis=0)
    
        if LED_i == LED_vec[0]:
            low_res_stack = low_res
        else:
            low_res_stack = tf.concat([low_res_stack,low_res],0)     
    
    
    return low_res_stack



def HiToLo_singleLED_multipupil_3D(obj_stack, H_prop_dz, H_prop_focal_plane, num_z_layers,\
                                   N_obj, N_patch, scale_multiply, \
                                   num_patches, Ns_mat, scale_mat, \
                                   cen0, P, Np, LED_i, change_pupil, \
                                   change_Ns_mat):

    low_res_patches=[]


    count = 0
    for i,startX in enumerate(np.arange(0,N_obj,N_patch)):
        for j,startY in enumerate(np.arange(0,N_obj,N_patch)):

                
            # pass the full object to HiToLoPatch
            Ns = Ns_mat[count,:,:]
            scale = scale_mat[count,:]
            
            if change_pupil:
                P_i = P[count,:,:]
            else:
                P_i = P
            
            low_res_patch_everything = HiToLoPatch_singleLED_multipupil_3D(obj_stack, H_prop_dz, H_prop_focal_plane, num_z_layers,\
                                                                           scale_multiply, Ns, scale, cen0, \
                                                                           P_i, Np, N_obj, LED_i, change_Ns_mat)
            low_res_patches.append(low_res_patch_everything)
            count += 1


    count = 0


    for i,startX in enumerate(np.arange(0,Np,Np/num_patches)):
        for j,startY in enumerate(np.arange(0,Np,Np/num_patches)):
            # Extract out patch of interest
            low_res_patch=tf.slice(low_res_patches[count],[int(startX),int(startY)],[int(N_patch*Np/N_obj),int(N_patch*Np/N_obj)])
            if j==0:
                low_res_obj_row=low_res_patch
            else:
                low_res_obj_row = tf.concat([low_res_obj_row,low_res_patch],axis=1)

            count += 1

        if i==0:
            low_res_obj = low_res_obj_row
        else:
            low_res_obj = tf.concat([low_res_obj,low_res_obj_row],axis=0)

    low_res_obj = tf.cast(low_res_obj,tf.float32)
    return low_res_obj



def HiToLoPatch_singleLED_multipupil_3D(obj_stack, H_prop_dz, H_prop_focal_plane, num_z_layers, scale_multiply, \
                                        Ns, scale, cen0, P, Np, N_obj, LED_i, change_Ns_mat):

    illumination_weight = scale[LED_i] * scale_multiply[LED_i]

    cen = (cen0-Ns[LED_i,:])
#    cen = (cen0-tf.cast(Ns[LED_i,:],tf.int32))

    O = F(obj_stack[0,:,:],N_obj,N_obj)
    O = shift_no_downsamp(O,cen,N_obj)
    
    for z in range(1,num_z_layers):
        
        # propagate
        O = O*H_prop_dz
        
        # multiply in real space and put back in Fourier space
        O = F(Ft(O,N_obj,N_obj)*obj_stack[z,:,:],N_obj,N_obj)
    
    #propage back to focal plane
    
    O = O*H_prop_focal_plane
    
    Psi0 = downsamp_slice(O,[N_obj//2,N_obj//2],Np,N_obj,change_Ns_mat)*P

    psi0 = Ft(Psi0,Np,Np) #low resolution field
    intensity_i = psi0*tf.conj(psi0)*tf.cast(illumination_weight, tf.complex64)

    return intensity_i

def shift_no_downsamp(O,cen,N_obj):
    
    #zero pad
    pad = N_obj//2
    O = tf.pad(O, [[int(pad),int(pad)],[int(pad),int(pad)]], 'CONSTANT')
    
    cen = cen + int(pad)
    
#    cen = tf.Print(cen,[cen])
    
    #slice
    O = tf.slice(O, [tf.cast(cen[0], tf.int32)-N_obj//2, \
                 tf.cast(cen[1], tf.int32)-N_obj//2], [N_obj, N_obj])
    
    return O

def downsamp_slice(x, cen, Np, N_obj, change_Ns_mat): 
    if change_Ns_mat:
        cen_floor = tf.floor(cen)
        cen_ceil = cen_floor + [1,1]
        cen_corner1 = cen_floor + [0,1]
        cen_corner2 = cen_floor + [1,0]
        
        slice_floor = tf.slice(x, [tf.cast(cen_floor[0], tf.int32)-Np//2, \
                                   tf.cast(cen_floor[1], tf.int32)-Np//2], [Np, Np])
        
        slice_ceil = tf.slice(x, [tf.cast(cen_ceil[0], tf.int32)-Np//2, \
                                   tf.cast(cen_ceil[1], tf.int32)-Np//2], [Np, Np])
    
    
        slice_corner1 = tf.slice(x, [tf.cast(cen_corner1[0], tf.int32)-Np//2, \
                                   tf.cast(cen_corner1[1], tf.int32)-Np//2], [Np, Np])
        
        slice_corner2 = tf.slice(x, [tf.cast(cen_corner2[0], tf.int32)-Np//2, \
                                   tf.cast(cen_corner2[1], tf.int32)-Np//2], [Np, Np])
        
        slice_average1 = slice_floor*tf.cast((cen_corner1[1] - cen[1]), tf.complex64) + slice_corner1*tf.cast((cen[1]-cen_floor[1]), tf.complex64)
        slice_average2 = slice_corner2*tf.cast((cen_ceil[1] - cen[1]),tf.complex64) + slice_ceil*tf.cast((cen[1]-cen_corner2[1]), tf.complex64)
        
        slice_average = slice_average1*tf.cast((cen_ceil[0]-cen[0]),tf.complex64) + slice_average2*tf.cast((cen[0]-cen_floor[0]),tf.complex64)
        
        return slice_average
    
    else:
        return tf.slice(x, [tf.cast(cen[0], tf.int32)-Np//2, \
                            tf.cast(cen[1], tf.int32)-Np//2], [Np, Np])


def CalculateParameters(N_patch_center, N_obj_center, dx_obj, hhled, ds_led, \
                        vvled, z_led, wavelength, du, LitCoord, numLEDs, NA, um_m):

    obj_center = (N_patch_center - N_obj_center)*dx_obj # [um] distance from N_obj_center to center of patch    

    # corresponding angles for each LEDs
    # this code assumes LED array and image are in the same orientation 
    dd = np.sqrt((-hhled*ds_led-obj_center[1])**2+(-vvled*ds_led-obj_center[0])**2+z_led**2);
    sin_thetah = (-hhled*ds_led-obj_center[1])/dd;
    sin_thetav = (-vvled*ds_led-obj_center[0])/dd;
    
    illumination_na = np.sqrt(sin_thetav**2+sin_thetah**2);
    
    ### corresponding spatial freq for each LEDs
    vled = sin_thetav/wavelength
    uled = sin_thetah/wavelength
    
    ### spatial freq index for each plane wave relative to the center
    idx_u = np.round(uled/du)
    idx_v = np.round(vled/du)
    
    dd_used=dd[np.nonzero(LitCoord)]
    illumination_na_used = illumination_na[np.nonzero(LitCoord)]
    
    # number of brightfield image LEDs
    NBF = len(np.nonzero(illumination_na_used<NA)[0])
    print('number of brightfield LEDs: ', NBF)
    
    # maxium spatial frequency achievable based on the maximum illumination
    # angle from the LED array and NA of the objective
    um_p = np.max(illumination_na_used)/wavelength+um_m
    
    synthetic_NA = um_p*wavelength
    print('synthetic NA is ', um_p*wavelength)
    
    # resolution achieved after freq post-processing
    dx0_p = 1./um_p/2.
    print('achieved resolution is: ', dx0_p)

    Ns=np.zeros([len(idx_u[np.nonzero(LitCoord)]),2]) #LED indices
    Ns[:,1]=idx_u[np.nonzero(LitCoord)] 
    Ns[:,0]=idx_v[np.nonzero(LitCoord)]
    scale = np.ones([numLEDs,])*(z_led/dd_used)
    
    return Ns, scale, synthetic_NA




def NAfilter(m,L,wavelength,NA):
    #m is the number of points in the source plane field (asuume square field)
    #L is the side length of the observation and source fields (assume square field)
    #wavelength is the free space wavelength

    dx=L/m
    k=1./wavelength #wavenumber #2*pi/wavelength #1./wavelength
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/L,m) #freq coords

    FX,FY=np.meshgrid(fx,fx)
    
    H=np.zeros([m,m])
    H[np.nonzero(np.sqrt(FX**2+FY**2)<=NA*k)]=1.


#    H=np.fft.fftshift(H)

    return H  