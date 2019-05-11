#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:20:38 2018

@author: vganapa1
"""
import numpy as np
import tensorflow as tf

def H_fresnel_prop(Nx,Ny,dx,dy,z,wavelength): # x is the row coordinate, y is the column coordinate

    Lx = Nx*dx
    Ly = Ny*dy
    
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Nx) #freq coords
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,Ny) #freq coords
    
    FX,FY=np.meshgrid(fx,fy, indexing = 'ij')

    H_fresnel=np.exp(-1j*np.pi*wavelength*z*(FX**2+FY**2))
    
    return H_fresnel


def create_H_fresnel_stack(Nx,Ny,dx,dy,z_vec,wavelength):
    H_fresnel_stack = np.zeros([Nx,Ny, len(z_vec)], dtype = np.complex64)
    for i,z in enumerate(z_vec):
        H_fresnel = H_fresnel_prop(Nx,Ny,dx,dy,z,wavelength)
        H_fresnel_stack[:,:,i] = H_fresnel 
    return H_fresnel_stack


def NA_filter(Nx,Ny,dx,dy,wavelength,NA):
    #wavelength is the free space wavelength
    
    Lx = Nx*dx
    Ly = Ny*dy

    k=1./wavelength #wave number 
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Nx) #freq coords
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,Ny) #freq coords

    FX,FY=np.meshgrid(fx,fy, indexing = 'ij')
    
    H_NA=np.zeros([Nx,Ny], dtype=np.complex64)
    H_NA[np.nonzero(np.sqrt(FX**2+FY**2)<=NA*k)]=1.

    return H_NA  

def apply_filter_function(u0,H,Nx,Ny,incoherent=False, library=tf):
    #u1 is the source plane field

    if incoherent:
        H=F(Ft(H,Nx,Ny,library)*library.conj(Ft(H,Nx,Ny,library)),Nx,Ny,library)

        U0=F(u0,Nx,Ny,library)

        U1=H*U0
        u1=Ft(U1,Nx,Ny,library)

    else:
        U0=F(u0,Nx,Ny,library)

        U1=H*U0
        u1=Ft(U1,Nx,Ny,library)

    return u1

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

#### Define Fourier and Inverse Fourier transform
    
def F(mat2D,dim0,dim1,library=tf):
    if library==tf:
        return fftshift(tf.fft2d(fftshift(mat2D, dim0, dim1)), dim0, dim1)
    elif library==np:
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mat2D)))

def Ft(mat2D,dim0,dim1,library=tf):
    if library==tf:
        return fftshift(tf.ifft2d(fftshift(mat2D, dim0, dim1)), dim0, dim1)
    elif library==np:
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(mat2D)))