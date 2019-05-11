#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:31:49 2018

@author: vganapa1
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

F = lambda x: np.fft.fftshift(np.fft.fft2(x))
Ft = lambda x: np.fft.ifft2(np.fft.ifftshift(x))

#F = lambda x: np.fft.fft2(np.fft.fftshift(x))
#Ft = lambda x: np.fft.ifftshift(np.fft.ifft2(x))

def scalar_prop(u0,dx,dy,z,wavelength): # x is the row coordinate, y is the column coordinate
    Nx = u0.shape[0]
    Ny = u0.shape[1]
    Lx = Nx*dx
    Ly = Ny*dy
    
    U0 = F(u0)
    
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Nx) #freq coords
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,Ny) #freq coords
    
    FX,FY=np.meshgrid(fx,fy, indexing = 'ij')

    H = np.exp(1j*2*math.pi*(1./wavelength)*z*np.sqrt(1-(wavelength*FX)**2+(wavelength*FY)**2))
    H[np.nonzero( np.sqrt(FX**2+FY**2) > (1./wavelength) )] = 0
    H=np.fft.fftshift(H)
    
    U1 = U0*H
    u1 = Ft(U1)
    
    return u1

def fresnel_prop(u0,dx,dy,z,wavelength): # x is the row coordinate, y is the column coordinate
    Nx = u0.shape[0]
    Ny = u0.shape[1]
    Lx = Nx*dx
    Ly = Ny*dy
    
    U0 = F(u0)
    
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Nx) #freq coords
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,Ny) #freq coords
    
    FX,FY=np.meshgrid(fx,fy, indexing = 'ij')

    H_fresnel=np.exp(-1j*math.pi*wavelength*z*(FX**2+FY**2))
    H_fresnel[np.nonzero( np.sqrt(FX**2+FY**2) > (1./wavelength) )] = 0
    H_fresnel=np.fft.fftshift(H_fresnel)
    
    U1 = U0*H_fresnel
    u1 = Ft(U1)
    
    return u1

def create_plane_wave(Nx, Ny, dx, dy, alpha, beta, wavelength):
    
    Lx = Nx*dx
    Ly = Ny*dy
    fx = alpha/wavelength
    fy = beta/wavelength
    
    x_vec=np.linspace(-Lx/2,Lx/2-dx,Nx)
    y_vec=np.linspace(-Ly/2,Ly/2-dy,Ny)     
    xm,ym=np.meshgrid(x_vec,y_vec, indexing = 'ij')
    
    plane_wave = np.exp(1j*2*math.pi*(fx*xm + fy*ym))
    return plane_wave


def create_plane_wave_3D(Nx, Ny, Nz, dx, dy, dz, alpha, beta, wavelength):
    
    gamma = np.sqrt(1 - alpha**2 - beta**2)
    
    Lx = Nx*dx
    Ly = Ny*dy
    Lz = Nz*dz
    
    fx = alpha/wavelength
    fy = beta/wavelength
    fz = gamma/wavelength
    
    x_vec=np.linspace(-Lx/2,Lx/2-dx,Nx)
    y_vec=np.linspace(-Ly/2,Ly/2-dy,Ny)
    z_vec=np.linspace(-Lz/2,Lz/2-dz,Nz)
    
    xm,ym,zm = np.meshgrid(x_vec,y_vec,z_vec, indexing = 'ij')
    
    plane_wave_3D = np.exp(1j*2*math.pi*(fx*xm + fy*ym + fz*zm))
    return plane_wave_3D

def create_circle(Nx,Ny,radius):

    x_vec=np.linspace(-Nx/2,Nx/2-1,Nx)
    y_vec=np.linspace(-Ny/2,Ny/2-1,Ny)
    xm,ym=np.meshgrid(x_vec,y_vec, indexing = 'ij')
    
    circle = np.zeros([Nx, Ny], dtype = np.complex64)

    circle[np.nonzero( np.sqrt(xm**2+ym**2) <= radius )] = 1
    return circle

def create_point(Nx,Ny,x,y):
    point = np.zeros([Nx, Ny], dtype = np.complex64)
    point[Nx/2 - x, Ny/2 - y] = 1
    return point

def create_rect(Nx, Ny, Nx_rect, Ny_rect):
    x_vec=np.linspace(-Nx/2,Nx/2-1,Nx)
    y_vec=np.linspace(-Ny/2,Ny/2-1,Ny)
    xm,ym=np.meshgrid(x_vec,y_vec, indexing = 'ij')        
    rect = np.ones([Nx, Ny], dtype = np.complex64)
    rect[(np.abs(xm) > Nx_rect)] = 0
    rect[(np.abs(ym) > Ny_rect)] = 0
    return rect

def CreateFig(mat,save_name,title='', vmin=None, vmax=None, cmap='viridis', angle=False): #folder_name is where the figure is saved
    
    if angle:
        mat[np.abs(mat-math.pi)<1e-10] = math.pi
        mat[np.abs(mat+math.pi)<1e-10] = math.pi
    
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(mat, interpolation='none', vmin=vmin, vmax=vmax) #cmap='gray'
    plt.colorbar()
    plt.savefig(save_name+'.png',bbox_inches='tight', dpi = 300) 

def fourier_transform_figs(u, save_name):
    
    u[np.abs(u)<1e-10] = 0
    
    CreateFig(np.abs(u), save_name + '_abs0', title = 'Magnitude of Function')
    CreateFig(np.angle(u), save_name + '_angle0', angle=True, title = 'Phase of Function')

    CreateFig(np.abs(F(u)), save_name + '_abs1', title = 'Magnitude of Fourier Transform')
    CreateFig(np.angle(F(u)), save_name + '_angle1', angle=True, title = 'Phase of Fourier Transform') 
    
#sys.exit()    

## 2D Fourier Transform examples

Nx = 2**8
Ny = 2**8
Nz = 2**8
dx = 1e-7
dy = 1e-7
dz = 1e-7

wavelength = 600e-9

circle = create_circle(Nx,Ny,2**0)
fourier_transform_figs(circle, 'circle')

circle = create_circle(Nx,Ny,2**5)
fourier_transform_figs(circle, 'circle')


rect = create_rect(Nx, Ny, 2**0, 2**0)
fourier_transform_figs(rect, 'rect')

rect = create_rect(Nx, Ny, 2**5, 2**5)
fourier_transform_figs(rect, 'rect')


plane_wave = create_plane_wave(Nx, Ny, dx, dy, 0, 0, wavelength)
fourier_transform_figs(plane_wave, 'plane_wave')

plane_wave = create_plane_wave(Nx, Ny, dx, dy, 0.6, 0.6, wavelength)
fourier_transform_figs(plane_wave, 'plane_wave')


fourier_transform_figs(circle*plane_wave, 'circle')

fourier_transform_figs(rect*plane_wave, 'rect')

## 3D plane wave

plane_wave_3D = create_plane_wave_3D(Nx, Ny, Nz, dx, dy, dz, 0.1, 0.1, wavelength)

CreateFig(np.angle(plane_wave_3D[:,:,0]),'plane_wave3D',title='Plane Wave z-Slice')
CreateFig(np.angle(plane_wave_3D[:,:,5]),'plane_wave3D',title='Plane Wave z-Slice')
CreateFig(np.angle(plane_wave_3D[:,:,10]),'plane_wave3D',title='Plane Wave z-Slice')
CreateFig(np.angle(plane_wave_3D[:,:,15]),'plane_wave3D',title='Plane Wave z-Slice')


### Light propagating

dx = 1e-6
dy = 1e-6
Nx = 2**8
Ny = 2**8
z = 20e-6
wavelength = 600e-9

u0 = create_point(Nx,Ny,0,0)
u1 = scalar_prop(u0,dx,dy,z,wavelength)
u2 = scalar_prop(u0,dx,dy,z*2,wavelength)
u3 = scalar_prop(u0,dx,dy,z*3,wavelength)

CreateFig(np.abs(u0),'u0',title='u0')
CreateFig(np.abs(u1),'u1',title='u1')
CreateFig(np.abs(u2),'u2',title='u2') 
CreateFig(np.abs(u3),'u3',title='u3') 

z = 200e-6
u0 = circle*plane_wave
u1 = scalar_prop(u0,dx,dy,z,wavelength)
CreateFig(np.abs(u0),'u0',title='u0')
CreateFig(np.abs(u1),'u1',title='u1')

