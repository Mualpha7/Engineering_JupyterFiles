#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:13:25 2018

@author: vganapa1
"""
import tensorflow as tf
import numpy as np

# fftshift implemented for Tensorflow
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

def F(mat2D,dim0,dim1): # Fourier transform
    return fftshift(tf.fft2d(fftshift(mat2D, dim0, dim1)), dim0, dim1)

def Ft(mat2D,dim0,dim1): # Inverse Fourier transform
    return fftshift(tf.ifft2d(fftshift(mat2D, dim0, dim1)), dim0, dim1)

def H_fresnel_prop(u0,Nx,Ny,dx,dy,z,wavelength): # x is the row coordinate, y is the column coordinate

    Lx = Nx*dx
    Ly = Ny*dy
    k=1./wavelength #wavenumber
    
    fx=tf.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Nx) #freq coords
    fy=tf.linspace(-1/(2*dy),1/(2*dy)-1/Ly,Ny) #freq coords
    
    FX,FY=tf.meshgrid(fx,fy, indexing = 'ij')
    FX_0 = tf.cast(FX,tf.complex64)
    FY_0 = tf.cast(FY,tf.complex64)
    H_fresnel=tf.exp(-1j*np.pi*tf.cast(wavelength,tf.complex64)*z*(FX_0**2+FY_0**2))
    
    zeros = tf.cast(tf.zeros([Nx,Ny]),dtype=tf.complex64)
    H_fresnel = tf.where(tf.sqrt(FX**2+FY**2)<=k,H_fresnel,zeros)
    
    return np.abs(Ft(F(u0,Nx,Ny)*H_fresnel,Nx,Ny))**2

# Dense neural network layer
def dense_layer(input_layer,Nx,Ny,output_size,name='dense_layer'):
    
    input_layer = tf.reshape(input_layer, [1,-1])
    
    W = tf.get_variable('W_dense', \
                       initializer = tf.truncated_normal([Nx*Ny, output_size], stddev=0.01), \
                       dtype = tf.float32)
    
    b = tf.get_variable('b_dense', \
                        initializer = tf.truncated_normal([output_size,], stddev=0.01), \
                        dtype = tf.float32)
      
    output_layer = tf.matmul(input_layer,W) + b 
    
    return output_layer

# Convolutional neural network layer
def conv_layer(kernel_length, input_layer, input_channels, output_channels):

    kernel = tf.get_variable('kernel', \
                             initializer = tf.truncated_normal([kernel_length,kernel_length,input_channels,output_channels], \
                                                            mean=0, stddev=0.1, dtype=tf.float32), \
                             dtype = tf.float32)
    biases = tf.get_variable('biases', \
                             initializer = tf.truncated_normal([output_channels], mean=0, stddev=0.01, dtype=tf.float32), \
                             dtype = tf.float32)
    conv_layer = tf.nn.conv2d(input_layer, kernel, [1, 1, 1, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, biases)
    conv_layer = tf.nn.relu(conv_layer)
    return conv_layer