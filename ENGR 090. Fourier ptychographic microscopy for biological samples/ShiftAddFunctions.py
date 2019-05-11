#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:58:54 2018

@author: vganapa1
"""
import numpy as np
import imageio
import math

def F(x): # Fourier transform
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def Ft(x): # Inverse Fourier transform
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

def shift(f,H):
    return np.real(Ft(F(f)*H))
    
# get derived parameters
def get_derived_params_SA(NA, wavelength, mag, dpix_c, Np, N_patch_center, N_img_center, lit_cenv, \
                       lit_cenh, ds_led, z_led, zmin, zmax, dz, dia_led):
    
    # maximum spatial frequency set by NA
#    um_m = NA/wavelength
    
    # system resolution based on the NA
#    dx0 = 1./um_m/2.
    
    # effective image pixel size on the object plane
    dpix_m = dpix_c/mag
    
    # FoV in the object space
#    FoV = Np*dpix_m;
    
    # set up spatial frequency coordinates
    umax = 1./2./dpix_m
    dv = 1./dpix_m/Np[0]
    du = 1./dpix_m/Np[1]
    u = np.arange(-umax, umax, du)
    v = np.arange(-umax, umax, dv)
    [u,v] = np.meshgrid(u,v)
    
    # set up image coordinates
#    nstart = ncent-Np/2
    
    # start pixel of the image patch
#    img_center = (nstart-ncent+Np/2)*dpix_m
    img_center = (N_patch_center - N_img_center)*dpix_m # [um] distance from N_img_center to center of patch   
    
    ### LED array derived quantities ###
    
    # set up LED coordinates
    # h: horizontal, v: vertical
    
    vled = np.arange(0,32,1)-lit_cenv
    hled = np.arange(0,32,1)-lit_cenh
    
    [hhled,vvled] = np.meshgrid(hled,vled)
    
#    dd = np.sqrt((-hhled*ds_led-img_center[0])**2+(-vvled*ds_led-img_center[1])**2+z_led**2)
#    sin_thetav = (-hhled*ds_led-img_center[0])/dd
#    sin_thetah = (-vvled*ds_led-img_center[1])/dd
    
    tan_thetav = (-hhled*ds_led-img_center[0])/z_led
    tan_thetah = (-vvled*ds_led-img_center[1])/z_led
    
    

    rrled = np.sqrt(hhled**2+vvled**2) # [LEDs] physical distance from center LED
    # Define coordinates in LED matrix that are actually used for imaging 
    LitCoord = np.zeros(hhled.shape)
    LitCoord[np.nonzero( rrled < (dia_led/2.) )] = 1. # LED array, with 1's for the LEDs actually used
    Litidx = np.nonzero(LitCoord) # index of LEDs used in the experiment
    Nimg = len(np.nonzero(LitCoord)[0]) # Total number of LEDs used in experiment    
    
    Tanv_lit = tan_thetav[Litidx]
    Tanh_lit = tan_thetah[Litidx]
    
    z = np.arange(zmin,zmax,dz)
    Nz = np.shape(z)[0]
    
    return u, v, Nz, z, Nimg, Tanh_lit, Tanv_lit

def read_images_SA(Np, Nimg, data_path):

    img_stack = np.zeros([Nimg, Np[0], Np[1]], dtype = np.complex64)
    
    for n in range(0, Nimg):
        img_address = 'Photo' + '0'*(4-len(str(n))) + str(n) + '.png'
        img = imageio.imread(data_path + img_address).astype(np.complex64)
        img_stack[n,:,:]=img

    return img_stack

def shift_add(img_stack, Np, Nz, z, Nimg, Tanh_lit, Tanv_lit, \
              u, v, all_mats=False):
    
    tot_mat = np.zeros([Nz, Np[0], Np[1]])
    
    if all_mats:
        left_mat = np.zeros([Nz, Np[0], Np[1]])
        right_mat = np.zeros([Nz, Np[0], Np[1]])
        top_mat = np.zeros([Nz, Np[0], Np[1]])
        bottom_mat = np.zeros([Nz, Np[0], Np[1]])
        DPC_lr_mat = np.zeros([Nz, Np[0], Np[1]])
        DPC_tb_mat = np.zeros([Nz, Np[0], Np[1]])
    
    

    for m in range(0, Nz):
        tot = np.zeros(Np)
        
        if all_mats:
            left = np.zeros(Np)
            right = np.zeros(Np)
            top = np.zeros(Np)
            bottom = np.zeros(Np)
        
    
        for n in range(0, Nimg):
    
            img = img_stack[n,:,:]
            
            ## shift
            # compute shift in Fourier shift for considering subpixel shift
            shift_x = z[m] * Tanh_lit[n]
            shift_y = z[m] * Tanv_lit[n]
            Hs = np.exp(1j*2*math.pi*(shift_x*u+shift_y*v))
            # shifted image
            img = shift(img,Hs)
            
            if all_mats:
                # add
                if Tanh_lit[n]>0:
                    left = left + img
                elif Tanh_lit[n]<0:
                    right = right + img
        
                if Tanv_lit[n]>0:
                    top = top + img
                elif Tanv_lit[n]<0:
                    bottom = bottom + img
            
            # refocused brightfield
            tot = tot + img
    
        tot_mat[m,:,:] = tot
        
        if all_mats:
            # computed refocused two-axis DPC
            # Left right DPC
            DPC_lr = (left-right)/tot
            # Top bottom DPC
            DPC_tb = (top-bottom)/tot
            
            left_mat[m,:,:] = left
            right_mat[m,:,:] = right
            top_mat[m,:,:] = top
            bottom_mat[m,:,:] = bottom
            DPC_lr_mat[m,:,:] = DPC_lr
            DPC_tb_mat[m,:,:] = DPC_tb
    
    if all_mats:
        return tot_mat, left_mat, right_mat, top_mat, bottom_mat, DPC_lr_mat, DPC_tb_mat
    else:
        return tot_mat



