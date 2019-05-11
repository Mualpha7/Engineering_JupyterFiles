# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:32:24 2017

@author: vidyag
"""

import tensorflow as tf
import numpy as np
import math
from scipy import misc
import glob

from autoencode import AutoEncoder
from layer_defs import variable_on_cpu, getConvInitializer


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


def propTF_withNA_PSFcustom(u1,H_fresnel,H_NA,PSF,m,incoherent=1):
    #u1 is the source plane field
    #L is the side length of the observation and source fields (assume square fields)

    #NA is the numerical aperture
    #m is u1.shape[0], is a tf.int32
    #dx is L/m

    PSF_phase=2*math.pi*PSF
    PSF_phase=tf.cast(tf.cos(PSF_phase),tf.complex64)-1j*tf.cast(tf.sin(PSF_phase),tf.complex64)

#    PSF_phase=2*tf.constant(math.pi, dtype=tf.complex64)*tf.cast(PSF, tf.complex64) #/tf.cast(tf.reduce_max(PSF), tf.complex64)
#    with tf.device('/cpu:0'):
#        PSF_phase=tf.exp(-1j*PSF_phase)

    PSF_phase=fftshift(PSF_phase,m,m)

    H = H_fresnel*H_NA*PSF_phase

    if incoherent:
        #H=np.fft.ifft2(np.abs(np.fft.fft2(H))**2)
        #H=tf.ifft2d(tf.fft2d(H)*tf.conj(tf.fft2d(H)))
        H=tf.fft2d(tf.ifft2d(H)*tf.conj(tf.ifft2d(H)))
        #H=H/H[0,0]

        U1=fftshift(u1,m,m)
        U1=tf.fft2d(U1)

        U2=H*U1
        u2=fftshift(tf.ifft2d(U2),m,m)

    else:
        U1=fftshift(u1,m,m)
        U1=tf.fft2d(U1)

        U2=H*U1
        u2=fftshift(tf.ifft2d(U2),m,m)
        u2 = u2*tf.conj(u2) # make into intensity object

    return u2


def change_filter_function(u1,H_old,H_new,Nx,reg=1e-10,incoherent=1):
    
    #u1 is the source plane field
    #Nx is u1.shape[0]
    #dx is L/m
    #H_old and H_new are already fftshifted in Fourier space


    if incoherent:
        H_old = tf.fft2d(tf.ifft2d(H_old)*tf.conj(tf.ifft2d(H_old)))
        H_new = tf.fft2d(tf.ifft2d(H_new)*tf.conj(tf.ifft2d(H_new)))
        #H=H/H[0,0]

        U1=fftshift(u1,Nx,Nx)
        U1=tf.fft2d(U1)

#        U2=H_old*U1 # previous processing
        U2 = (H_new/(H_old+reg))*U1

        
        u2=fftshift(tf.ifft2d(U2),Nx,Nx)

    else:
        U1=fftshift(u1,Nx,Nx)
        U1=tf.fft2d(U1)

        U2=(H_new/(H_old+reg))*U1

#        U2=H_old*U1 # previous processing
        u2=fftshift(tf.ifft2d(U2),Nx,Nx) # field value
#        u2 = u2*np.conj(u2) # make into intensity object

    return u2


def create_phase_obj_stack(PSF, trainingSet, batch_size, H_fresnel, H_NA, Nx):
    for ii in range(batch_size):

        # low_res_obj is an intensity object
        low_res_obj = propTF_withNA_PSFcustom(trainingSet[ii,:,:],H_fresnel,H_NA,PSF,Nx,incoherent=0)

        low_res_obj = tf.expand_dims(low_res_obj,axis=0)

        if ii == 0:
            low_res_obj_stack = low_res_obj
        else:
            low_res_obj_stack = tf.concat([low_res_obj_stack,low_res_obj],0)

    low_res_obj_stack = tf.cast(low_res_obj_stack, tf.float32)

    return low_res_obj_stack


def create_microscope_img(PSF,trainingSet_sample,Nz,H_fresnel_stack,H_NA,m,num_wavelengths):
    # trainingSet_sample = trainingSet[ii,:,:,:]

    microscopeImg=[]

    for zInd in range(Nz):
        for waveInd in range(num_wavelengths):
            add_layer=propTF_withNA_PSFcustom(trainingSet_sample[:,:,zInd,waveInd],H_fresnel_stack[:,:,zInd,waveInd],H_NA[:,:,waveInd],PSF,m)
            microscopeImg.append(add_layer)

    microscopeImg = tf.add_n(microscopeImg)
    microscopeImg = tf.expand_dims(microscopeImg,axis=0)
    microscopeImg = tf.cast(microscopeImg, tf.float32)
    return microscopeImg


def create_microscopeImgStack(PSF,trainingSet,Nz,batch_size,H_fresnel_stack,H_NA,m,num_wavelengths): #creates microscopeImg for every example in the trainingSet
    
    for ii in range(batch_size):
        microscopeImg = create_microscope_img(PSF,trainingSet[ii,:,:,:,:],Nz,H_fresnel_stack,H_NA,m,num_wavelengths)
        if ii == 0:
            microscopeImgStack = microscopeImg
        else:
            microscopeImgStack = tf.concat([microscopeImgStack,microscopeImg],0)

    return microscopeImgStack


def add_noise_microscopeImgStack(microscopeImgStack,normalRandomMat1,normalRandomMat2,sqrt_reg,\
                                 poisson_noise_multiplier, gaussian_noise_multiplier, batch_size, library=tf):

    #XXX Fix the Poisson noise for low photon levels
    
    if library == tf:
        multiplierPoisson = tf.constant(poisson_noise_multiplier,dtype=tf.float32) #6e3 for EPFL
        multiplierGaussian = tf.constant(gaussian_noise_multiplier,dtype=tf.float32)
    else:
        multiplierPoisson = poisson_noise_multiplier
        multiplierGaussian = gaussian_noise_multiplier
    
    microscopeImgStack2 = microscopeImgStack*multiplierPoisson

    microscopeImgStack3=library.sqrt(library.abs(microscopeImgStack2)+sqrt_reg)*normalRandomMat1+microscopeImgStack2
    microscopeImgStack4=microscopeImgStack3+multiplierGaussian*normalRandomMat2

    
    zeros = library.zeros([batch_size, microscopeImgStack4.shape[1], microscopeImgStack4.shape[2]], dtype=library.float32)

    microscopeImgStack = library.where(microscopeImgStack4<0,zeros,microscopeImgStack4) #truncate below 0
    
    microscopeImgStack = microscopeImgStack/multiplierPoisson
    return microscopeImgStack


def F(mat2D,dim0,dim1):
    return fftshift(tf.fft2d(mat2D),dim0,dim1)

def Ft(mat2D,dim0,dim1):
    return tf.ifft2d(fftshift(mat2D,dim0,dim1))

downsamp = lambda x,cen,Np:  x[cen[0]-Np//2:cen[0]-Np//2+Np, \
    cen[1]-Np//2:cen[1]-Np//2+Np]


def sigmoid_stretch(x, stretch, library=tf):
    y = 1. / (1. + library.exp(-x/stretch))
    return y


def HiToLoPatch_singleLED(obj,scale_multiply, Ns, scale, cen0, P, Np, N_obj, LED_i): #stretch

    illumination_weight = scale[LED_i] * scale_multiply[LED_i]


    cen = (cen0-Ns[LED_i,:]).astype(int) 
    O=F(obj,N_obj,N_obj)

    Psi0 = downsamp(O,cen,Np)*P

    psi0 = Ft(Psi0,Np,Np) #low resolution field
    intensity_i = psi0*tf.conj(psi0)*tf.cast(illumination_weight, tf.complex64)

    return intensity_i


def HiToLo_singleLED(obj, N_obj, N_patch, scale_multiply, num_patches, Ns_mat, scale_mat, \
           cen0, P, Np, LED_i):

    low_res_patches=[]


    count = 0
    for i,startX in enumerate(np.arange(0,N_obj,N_patch)):
        for j,startY in enumerate(np.arange(0,N_obj,N_patch)):

            # pass the full object to HiToLoPatch
            Ns = Ns_mat[count,:,:]
            scale = scale_mat[count,:]
            low_res_patch_everything = HiToLoPatch_singleLED(obj,scale_multiply, Ns, scale, cen0, P, Np, N_obj, LED_i)
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



def HiToLoPatch(obj,scale_multiply, Ns, scale, cen0, P, H0, Np, N_obj, numLEDs): #stretch

    illumination_weights = scale * scale_multiply #sigmoid_stretch(scale_multiply,stretch) #scale_multiply

    for LED_i in range(numLEDs):
        cen = (cen0-Ns[LED_i,:]).astype(int)
        O=F(obj,N_obj,N_obj)

        Psi0 = downsamp(O*H0,cen,Np)*P

        psi0 = Ft(Psi0,Np,Np) #low resolution field
        intensity_i = psi0*tf.conj(psi0)*tf.cast(illumination_weights[LED_i], tf.complex64)

        if LED_i == 0:
            low_res_patch = intensity_i
        else:
            low_res_patch = low_res_patch + intensity_i

    return low_res_patch

def HiToLo(obj, N_obj, N_patch, scale_multiply, num_patches, Ns_mat, scale_mat, \
           cen0, P, H0, Np, numLEDs): #stretch

    low_res_patches=[]


    count = 0
    for i,startX in enumerate(np.arange(0,N_obj,N_patch)):
        for j,startY in enumerate(np.arange(0,N_obj,N_patch)):

            # pass the full object to HiToLoPatch
            Ns = Ns_mat[count,:,:]
            scale = scale_mat[count,:]
            low_res_patch_everything = HiToLoPatch(obj,scale_multiply, Ns, scale, cen0, P, H0, Np, N_obj, numLEDs) #stretch
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

    return low_res_obj

def upsample(low_res_obj, Np, N_obj):
    
    if Np == 1:
        dense_multiply = tf.cast(tf.squeeze(getConvInitializer(N_obj, N_obj, init_type="trunc_norm")), tf.complex64)
        dense_bias = tf.cast(tf.squeeze(getConvInitializer(N_obj, N_obj, init_type="trunc_norm")), tf.complex64)
        
#        dense_multiply_i = tf.cast(tf.squeeze(getConvInitializer(N_obj, N_obj, init_type="trunc_norm")), tf.complex64)
#        dense_bias_i = tf.cast(tf.squeeze(getConvInitializer(N_obj, N_obj, init_type="trunc_norm")), tf.complex64)
        
        
#        pad0 = (N_obj - Np)//2 -1
#        pad1 = pad0 + 3
#        upsampled_obj = tf.pad(upsampled_obj, [[int(pad0),int(pad1)],[int(pad0),int(pad1)]], 'CONSTANT', \
#                                                constant_values = 100)
#        upsampled_obj_real = low_res_obj*dense_multiply_r + dense_bias_r
#        upsampled_obj_imag = low_res_obj*dense_multiply_i + dense_bias_i
        upsampled_obj = low_res_obj*dense_multiply + dense_bias
#        upsampled_obj = tf.Print(upsampled_obj, [dense_multiply], message='dense_multiply: ')
#        upsampled_obj = tf.Print(upsampled_obj, [dense_bias], message='dense_bias: ')
#        upsampled_obj = tf.Print(upsampled_obj, [low_res_obj], message='low_res_obj: ')

    else:    
        upsampled_obj = F(low_res_obj,Np,Np)
        pad = (N_obj - Np)/2
        upsampled_obj = tf.pad(upsampled_obj, [[int(pad),int(pad)],[int(pad),int(pad)]], 'CONSTANT')
        upsampled_obj = Ft(upsampled_obj,N_obj,N_obj)

    return upsampled_obj

def create_upsampled_obj_stack(low_res_obj_stack, batch_size, Np, N_obj):
    low_res_obj_stack = tf.cast(low_res_obj_stack, tf.complex64)
    for ii in range(batch_size):
        upsampled_obj = upsample(low_res_obj_stack[ii,:,:], Np, N_obj)
        upsampled_obj = tf.expand_dims(upsampled_obj,axis=0)
        if ii == 0:
            upsampled_obj_stack = upsampled_obj
        else:
            upsampled_obj_stack = tf.concat([upsampled_obj_stack,upsampled_obj],0)

    upsampled_obj_stack = tf.cast(upsampled_obj_stack, tf.float32)
    return upsampled_obj_stack

def create_FP_img_stack(scale_multiply,trainingSet,batch_size, N_obj, N_patch, num_patches, Ns_mat, \
                        scale_mat, cen0, P, H0, Np, numLEDs): 
    for ii in range(batch_size):
        low_res_obj = HiToLo(trainingSet[ii,:,:], N_obj, N_patch, scale_multiply, \
                             num_patches, Ns_mat, scale_mat, \
                             cen0, P, H0, Np, numLEDs)

        low_res_obj = tf.expand_dims(low_res_obj,axis=0)

        if ii == 0:
            low_res_obj_stack = low_res_obj
        else:
            low_res_obj_stack = tf.concat([low_res_obj_stack,low_res_obj],0)

    low_res_obj_stack = tf.cast(low_res_obj_stack, tf.float32)
    return low_res_obj_stack





def calculate_loss_FP(predicted_mat, trainingSet, library = tf):

    if library == tf:
        sum_func = tf.reduce_sum
    else:
        sum_func = np.sum
        
    loss_l2 = sum_func(library.real((predicted_mat - trainingSet)*library.conj(predicted_mat - trainingSet)))
#    loss_l1 = sum_func(library.abs(predicted_mat - trainingSet))

    return loss_l2


def grad_diff_loss(predicted_mat, trainingSet, library = tf):

    if library == tf:
        sum_func = tf.reduce_sum
    else:
        sum_func = np.sum
    
    diff_x_actual = trainingSet[:,1:,:]-trainingSet[:,:-1,:]
    diff_y_actual = trainingSet[:,:,1:]-trainingSet[:,:,:-1]

    diff_x_guess = predicted_mat[:,1:,:]-predicted_mat[:,:-1,:]
    diff_y_guess= predicted_mat[:,:,1:]-predicted_mat[:,:,:-1]

    loss_x = sum_func(library.real((diff_x_actual-diff_x_guess)*library.conj(diff_x_actual-diff_x_guess)))
    loss_y = sum_func(library.real((diff_y_actual-diff_y_guess)*library.conj(diff_y_actual-diff_y_guess)))

    return loss_x + loss_y

def convert_net_prediction_list(net_prediction_list, image_modality, batch_size, optical_parameters_dict):
    if (image_modality == 'FP') or (image_modality == 'phase') or (image_modality == 'FP_PP'):

        predicted_mat = tf.cast(net_prediction_list[0],tf.complex64)+1j*tf.cast(net_prediction_list[1],tf.complex64)
#        predicted_mat = tf.cast(net_prediction_list[0],tf.complex64)*tf.exp(1j*tf.cast(net_prediction_list[1],tf.complex64))

        predicted_mat = tf.squeeze(predicted_mat,axis=3)

    elif image_modality == 'STORM':
        Nx = optical_parameters_dict['Nx_highres']
        Nz = optical_parameters_dict['Nz']
        num_wavelengths = optical_parameters_dict['num_wavelengths']
        
        predicted_mat = tf.stack(net_prediction_list,axis=3)
        predicted_mat = tf.reshape(predicted_mat, [batch_size, Nx, Nx, Nz, num_wavelengths])
        
    elif image_modality == 'HE':         
        predicted_mat = tf.stack(net_prediction_list,axis=3)

        predicted_mat = tf.squeeze(predicted_mat, axis = 4)


    return predicted_mat

def iterative_solver_FP(predicted_mat, optical_element, batch_size, N_obj, N_patch, num_patches, Ns_mat,\
                        scale_mat, cen0, P, H0, Np, numLEDs, low_res_obj_stack, step_size, max_internal_iter, \
                        merit_stopping_point, loss_low_res, i):

#    print 'mii2'
    predicted_mat_real = tf.real(predicted_mat)
    predicted_mat_imag = tf.imag(predicted_mat)

    def single_iter(predicted_mat_real, predicted_mat_imag, loss_low_res, i):

        predicted_mat = tf.cast(predicted_mat_real,tf.complex64)+1j*tf.cast(predicted_mat_imag,tf.complex64)
        low_res_obj_predicted = create_FP_img_stack(optical_element,predicted_mat,batch_size, N_obj, N_patch, num_patches, Ns_mat, \
                        scale_mat, cen0, P, H0, Np, numLEDs) #stretch

        loss_low_res = tf.reduce_sum(tf.square(low_res_obj_stack - low_res_obj_predicted))

#        loss_MSE = calculate_loss_FP(low_res_obj_predicted, low_res_obj_stack)
#        loss_grad = grad_diff_loss(low_res_obj_predicted, low_res_obj_stack)
        

        #find gradient of loss_low_res with respect to predicted_mat
        predicted_mat_real_gradient = tf.squeeze(tf.gradients(loss_low_res,predicted_mat_real),axis=0)
        predicted_mat_imag_gradient = tf.squeeze(tf.gradients(loss_low_res,predicted_mat_imag),axis=0)

#        predicted_mat_real_gradient = tf.Print(predicted_mat_real_gradient, [loss_low_res], message='loss_low_res: ')

        #update predicted_mat with step_size
        predicted_mat_real = predicted_mat_real - step_size*predicted_mat_real_gradient
        predicted_mat_imag = predicted_mat_imag - step_size*predicted_mat_imag_gradient

        i+=1

        return predicted_mat_real, predicted_mat_imag, loss_low_res, i

    def convergence_cond(predicted_mat_real, predicted_mat_imag, loss_low_res,i):
        result = True
        
        
#        result = tf.Print(result, [result], message='result1: ')
       
        result = tf.cond(i>=max_internal_iter, lambda: False, lambda: result)
        
#        result = tf.Print(result, [result], message='result2: ')
        
        result = tf.cond(loss_low_res<merit_stopping_point, lambda: False, lambda: result)
        
#        result = tf.Print(result, [result], message='result3: ')
        
        return result


    [predicted_mat_real, predicted_mat_imag, loss_low_res, i] = tf.while_loop(convergence_cond,
                                                                              single_iter,
                                                                              [predicted_mat_real, predicted_mat_imag, loss_low_res,i])


    predicted_mat = tf.cast(predicted_mat_real,tf.complex64)+1j*tf.cast(predicted_mat_imag,tf.complex64)

    return predicted_mat

def iterative_solver_STORM(predicted_mat, PSF, Nz,batch_size, H_fresnel_stack, \
                           H_NA, Nx, low_res_obj_stack, step_size, max_internal_iter, \
                           merit_stopping_point, loss_low_res, i, num_wavelengths):

    def single_iter(predicted_mat, loss_low_res, i):

        predicted_mat_complex=tf.cast(predicted_mat, tf.complex64)
        low_res_obj_predicted = create_microscopeImgStack(PSF,predicted_mat_complex,Nz,batch_size,H_fresnel_stack,H_NA,Nx,num_wavelengths)

        loss_low_res = tf.reduce_sum(tf.square(low_res_obj_stack - low_res_obj_predicted))

        #find gradient of loss_low_res with respect to predicted_mat
        predicted_mat_gradient = tf.squeeze(tf.gradients(loss_low_res,predicted_mat),axis=0)

    #            predicted_mat_gradient = tf.Print(predicted_mat_gradient, [loss_low_res], message='loss_low_res: ')

        #update predicted_mat with step_size
        predicted_mat = predicted_mat - step_size*predicted_mat_gradient

        i+=1

        return predicted_mat, loss_low_res, i

    def convergence_cond(predicted_mat, loss_low_res,i):
        result = True
        result = tf.cond(i>=max_internal_iter, lambda: False, lambda: result)
        result = tf.cond(loss_low_res<merit_stopping_point, lambda: False, lambda: result)
        return result


    [predicted_mat, loss_low_res, i] = tf.while_loop(convergence_cond,
                                                    single_iter,
                                                    [predicted_mat, loss_low_res,i])
    return predicted_mat


def iterative_solver_phase(predicted_mat, PSF, batch_size, H_fresnel, \
                           H_NA, Nx, low_res_obj_stack, step_size, max_internal_iter, \
                           merit_stopping_point, loss_low_res, i):

    predicted_mat_real = tf.real(predicted_mat)
    predicted_mat_imag = tf.imag(predicted_mat)

    def single_iter(predicted_mat_real, predicted_mat_imag, loss_low_res, i):

        predicted_mat = tf.cast(predicted_mat_real,tf.complex64)+1j*tf.cast(predicted_mat_imag,tf.complex64)
        low_res_obj_predicted = create_phase_obj_stack(PSF, predicted_mat, batch_size, H_fresnel, H_NA, Nx)

        loss_low_res = tf.reduce_sum(tf.square(low_res_obj_stack - low_res_obj_predicted))

        #find gradient of loss_low_res with respect to predicted_mat
        predicted_mat_real_gradient = tf.squeeze(tf.gradients(loss_low_res,predicted_mat_real),axis=0)
        predicted_mat_imag_gradient = tf.squeeze(tf.gradients(loss_low_res,predicted_mat_imag),axis=0)

        #update predicted_mat with step_size
        predicted_mat_real = predicted_mat_real - step_size*predicted_mat_real_gradient
        predicted_mat_imag = predicted_mat_imag - step_size*predicted_mat_imag_gradient

        i+=1

        return predicted_mat_real, predicted_mat_imag, loss_low_res, i

    def convergence_cond(predicted_mat_real, predicted_mat_imag, loss_low_res,i):
        result = True
        result = tf.cond(i>=max_internal_iter, lambda: False, lambda: result)
        result = tf.cond(loss_low_res<merit_stopping_point, lambda: False, lambda: result)
        return result


    [predicted_mat_real, predicted_mat_imag, loss_low_res, i] = tf.while_loop(convergence_cond,
                                                                              single_iter,
                                                                              [predicted_mat_real, predicted_mat_imag, loss_low_res,i])


    predicted_mat = tf.cast(predicted_mat_real,tf.complex64)+1j*tf.cast(predicted_mat_imag,tf.complex64)


    return predicted_mat


def iterative_solver_HE(predicted_mat, Nx_highres, h_blur, \
                       high_magn, low_magn, dpix_c, wavelength, \
                       low_NA, low_res_obj_batch, step_size, max_internal_iter, \
                       merit_stopping_point, loss_low_res, i, \
                       batch_size):

    def single_iter(predicted_mat, loss_low_res, i):

        low_res_obj_batch_predicted = change_magn_batch(predicted_mat, Nx_highres, h_blur, \
                                                        high_magn, low_magn, dpix_c, wavelength, low_NA, batch_size)



        low_res_obj_batch_predicted = tf.cast(low_res_obj_batch_predicted, tf.float32)

        loss_low_res = tf.reduce_sum(tf.square(low_res_obj_batch - low_res_obj_batch_predicted))

        #find gradient of loss_low_res with respect to predicted_mat

        predicted_mat_gradient = tf.squeeze(tf.gradients(loss_low_res,predicted_mat),axis=0)

#            predicted_mat_gradient = tf.Print(predicted_mat_gradient, [predicted_mat_gradient], message='predicted_mat_gradient: ')
#            predicted_mat_gradient = tf.Print(predicted_mat_gradient, [loss_low_res], message='loss_low_res: ')

        #update predicted_mat with step_size
        predicted_mat = predicted_mat - step_size*predicted_mat_gradient

        i+=1

        return predicted_mat, loss_low_res, i

    def convergence_cond(predicted_mat, loss_low_res,i):
        result = True
        result = tf.cond(i>=max_internal_iter, lambda: False, lambda: result)
        result = tf.cond(loss_low_res<merit_stopping_point, lambda: False, lambda: result)
        return result


    [predicted_mat, loss_low_res, i] = tf.while_loop(convergence_cond,
                                                     single_iter,
                                                     [predicted_mat, loss_low_res,i])


    return predicted_mat, loss_low_res


def find_predicted_mat(image_modality, num_nets, input_layer_list, Nx, batch_size, layers_dropout, dropout_prob,
                       use_batch_norm, autoencode_init_type, init_type_bias, init_type_resid, kernel_multiplier, variance_reg, training, \
                       num_layers_autoencode, skip_interval, num_blocks, optical_parameters_dict):

    net_prediction_list = []
    for net in range(num_nets):
        with tf.variable_scope("net_" + str(net)):

            curr_input = input_layer_list[net]
            for i in range(num_blocks):
                with tf.variable_scope("ae_" + str(i)):
                    curr_net = AutoEncoder(curr_input, 
                                           num_layers_autoencode, Nx,
                                           batch_size, training,
                                           kernel_multiplier, 
                                           skip_interval=skip_interval, 
                                           conv_activation='maxout',
                                           deconv_activation='maxout',
                                           dropout_count=layers_dropout, 
                                           dropout_prob=dropout_prob,
                                           create_graph_viz=False, 
                                           use_batch_norm=use_batch_norm,
                                           init_type=autoencode_init_type, 
                                           init_type_bias=init_type_bias, 
                                           init_type_resid=init_type_resid)
                    curr_input = curr_net.get_prediction()
            net0_prediction = curr_input
            net_prediction_list.append(net0_prediction)
            
            """
            with tf.variable_scope("ae_0"):
                net_init = AutoEncoder(input_layer_list[net], num_layers_autoencode, Nx, batch_size, training, \
                                           kernel_multiplier, skip_interval=skip_interval, conv_activation='maxout', \
                                           deconv_activation='maxout', dropout_count=layers_dropout, dropout_prob=dropout_prob, \
                                           create_graph_viz=False, use_batch_norm=use_batch_norm, \
                                           init_type=autoencode_init_type, init_type_bias=init_type_bias, init_type_resid=init_type_resid)
                net_init_prediction = net_init.get_prediction()

            with tf.variable_scope("ae_1"):
                net0 = AutoEncoder(net_init_prediction, num_layers_autoencode, Nx, batch_size, training, \
                                       kernel_multiplier, skip_interval=skip_interval, conv_activation='maxout', \
                                       deconv_activation='maxout', dropout_count=layers_dropout, dropout_prob=dropout_prob, \
                                       create_graph_viz=False, use_batch_norm=use_batch_norm, \
                                       init_type=autoencode_init_type, init_type_bias=init_type_bias, init_type_resid=init_type_resid)
                net0_prediction = net0.get_prediction()
                net_prediction_list.append(net0_prediction)
            """
                
    ### End Neural Network(s)

    ### Convert net_prediction_list to predicted_mat

    predicted_mat = convert_net_prediction_list(net_prediction_list, image_modality, batch_size, optical_parameters_dict)

    return predicted_mat


def tower_loss_all(trainingSet, training, normalRandomMat1, normalRandomMat2, \
               initialize_optical_element_ones, num_elements, use_batch_norm, variance_reg, add_noise, sqrt_reg, \
               dropout_prob, layers_dropout, batch_size, max_internal_iter, merit_stopping_point, optical_parameters_dict, \
               autoencode_init_type, init_type_bias, init_type_resid, \
               kernel_multiplier, \
               poisson_noise_multiplier, gaussian_noise_multiplier, image_modality,
               num_layers_autoencode, skip_interval, num_blocks, \
               training_data_folder, load_optical_element_init,
               lowres_trainingSet=None):

    with tf.variable_scope("optical_transform"):

        Nx = optical_parameters_dict['Nx_highres']
        Nz = optical_parameters_dict['Nz']

        if initialize_optical_element_ones:
            if (image_modality == 'FP') or (image_modality == 'FP_PP'):
                optical_element0 = np.ones([num_elements,], dtype=np.float32)
            elif (image_modality == 'STORM') or (image_modality == 'phase'):
                optical_element0 = np.zeros([num_elements,], dtype=np.float32)
                optical_element0[0] = 1
        elif load_optical_element_init:
            optical_element0 = np.load(training_data_folder + '/optical_element_init.npy')
            
        else:
            optical_element0 = np.random.rand(num_elements,).astype(np.float32)

        if image_modality == 'FP':

            optical_element = variable_on_cpu('optical_element', optical_element0, tf.float32, \
                                              constraint = lambda x: tf.clip_by_value(x, 0, 1.0))


#            optical_element = variable_on_cpu('optical_element', optical_element0, tf.float32) #stretch


            num_nets = 2 # one neural net for real and one for imaginary

            numLEDs = optical_parameters_dict['numLEDs']
            N_obj = optical_parameters_dict['N_obj']
            N_patch = optical_parameters_dict['N_patch']
            num_patches = optical_parameters_dict['num_patches']
            Ns_mat = optical_parameters_dict['Ns_mat']
            scale_mat = optical_parameters_dict['scale_mat']
            cen0 = optical_parameters_dict['cen0']
            P = optical_parameters_dict['P']
            H0 = optical_parameters_dict['H0']
            Np = optical_parameters_dict['Np']

#            stretch = variable_on_cpu('sigmoid_stretch', 1.0, tf.float32, trainable=False)

            ### Convert trainingSet to low resolution

            low_res_obj_stack = create_FP_img_stack(optical_element,trainingSet,batch_size, N_obj, N_patch, num_patches, Ns_mat, \
                            scale_mat, cen0, P, H0, Np, numLEDs) #stretch

        elif image_modality == 'FP_PP':
            
            numLEDs = optical_parameters_dict['numLEDs']
            N_obj = optical_parameters_dict['N_obj']
            Np = optical_parameters_dict['Np']
            N_patch = optical_parameters_dict['N_patch']
            num_patches = optical_parameters_dict['num_patches']
            Ns_mat = optical_parameters_dict['Ns_mat']
            scale_mat = optical_parameters_dict['scale_mat']
            cen0 = optical_parameters_dict['cen0']
            P = optical_parameters_dict['P']
            H0 = optical_parameters_dict['H0']        
            
            optical_element = variable_on_cpu('optical_element', optical_element0, tf.float32, \
                                              constraint = lambda x: tf.clip_by_value(x, 0, 1.0))
                        
            num_nets = 2 # one neural net for real and one for imaginary
            
            low_res_obj_stack = lowres_trainingSet*optical_element
            low_res_obj_stack = tf.reduce_sum(low_res_obj_stack, axis=-1)
            
        elif image_modality == 'STORM':

            optical_element = variable_on_cpu('optical_element', optical_element0, tf.float32)

            num_wavelengths = optical_parameters_dict['num_wavelengths']
            num_nets = Nz*num_wavelengths

            H_fresnel_stack = optical_parameters_dict['H_fresnel_stack']
            H_NA = optical_parameters_dict['H_NA']
            ZernikePolyMat = optical_parameters_dict['ZernikePolyMat']


            PSF = ZernikePolyMat*optical_element
            PSF = tf.reduce_sum(PSF,axis=2)

            # Process 3D matrices to create the two-dimensional image

            low_res_obj_stack = create_microscopeImgStack(PSF, tf.cast(trainingSet, tf.complex64), Nz, batch_size, H_fresnel_stack, H_NA, Nx, num_wavelengths)

        elif image_modality == 'phase':

            optical_element = variable_on_cpu('optical_element', optical_element0, tf.float32)
            num_nets = 2 # one neural net for real and one for imaginary

            H_fresnel = optical_parameters_dict['H_fresnel']
            H_NA = optical_parameters_dict['H_NA']
            ZernikePolyMat = optical_parameters_dict['ZernikePolyMat']

            PSF = ZernikePolyMat*optical_element
            PSF = tf.reduce_sum(PSF,axis=2)

            low_res_obj_stack = create_phase_obj_stack(PSF, trainingSet, batch_size, H_fresnel, H_NA, Nx)



        low_res_obj_stack_nonoise = low_res_obj_stack

        if add_noise:
            low_res_obj_stack=add_noise_microscopeImgStack(low_res_obj_stack, normalRandomMat1, normalRandomMat2,\
                                                           sqrt_reg, poisson_noise_multiplier, gaussian_noise_multiplier, \
                                                           batch_size)


        if (image_modality == 'FP') or (image_modality == 'FP_PP'):
            # upsample the low_res_obj_stack for FP modality
            input_layer = create_upsampled_obj_stack(low_res_obj_stack, batch_size, Np, N_obj)
            input_layer = tf.expand_dims(input_layer, axis=3)
#            input_layer = tf.sqrt( (input_layer + sqrt_reg) / numLEDs)

        elif (image_modality == 'STORM') or (image_modality == 'phase'):
            input_layer = tf.expand_dims(low_res_obj_stack, axis=3)

    ### Neural Network(s)

    with tf.variable_scope("neural_net"):
        
        input_layer_list = [input_layer for i in range(num_nets)]    
        predicted_mat = find_predicted_mat(image_modality, num_nets, input_layer_list, Nx, batch_size, layers_dropout, dropout_prob, \
                               use_batch_norm, autoencode_init_type, init_type_bias, init_type_resid, kernel_multiplier, variance_reg, training, \
                               num_layers_autoencode, skip_interval, num_blocks, optical_parameters_dict)

        if (image_modality == 'FP_PP') and (max_internal_iter == 0):
            
            loss = calculate_loss_FP(predicted_mat, trainingSet)
            
        else:    
            ### convert predicted_mat down to low_res_obj_predicted
            if (image_modality == 'FP') or (image_modality == 'FP_PP'):
    
                low_res_obj_predicted = create_FP_img_stack(optical_element,predicted_mat,batch_size, N_obj, N_patch, num_patches, Ns_mat, \
                                scale_mat, cen0, P, H0, Np, numLEDs) #stretch
    
            elif image_modality == 'STORM':
    
                predicted_mat_complex=tf.cast(predicted_mat, tf.complex64)
                low_res_obj_predicted = create_microscopeImgStack(PSF,predicted_mat_complex,Nz,batch_size,H_fresnel_stack,H_NA,Nx,num_wavelengths)
    
            elif image_modality == 'phase':
    
                low_res_obj_predicted = create_phase_obj_stack(PSF, predicted_mat, batch_size, H_fresnel, H_NA, Nx)
    
            loss_low_res = tf.reduce_sum(tf.square(low_res_obj_stack - low_res_obj_predicted))
    
    ###################
            step_size = variable_on_cpu('step_size', 1e-7, tf.float32, trainable = False) # XXX make step_size trainable
    
        
            i = tf.Variable(0, dtype=tf.int32, trainable=False)

            if (image_modality == 'FP') or (image_modality == 'FP_PP'):
#                print 'mii4'
                if max_internal_iter > 0:    
#                    print 'mii3'
                    predicted_mat = iterative_solver_FP(predicted_mat, optical_element, batch_size, N_obj, N_patch, num_patches, Ns_mat,\
                                scale_mat, cen0, P, H0, Np, numLEDs, low_res_obj_stack, step_size, max_internal_iter, \
                                merit_stopping_point, loss_low_res, i)
    
                loss = calculate_loss_FP(predicted_mat, trainingSet)
    
            elif image_modality == 'STORM':
                if max_internal_iter > 0:
                    predicted_mat = iterative_solver_STORM(predicted_mat, PSF, Nz,batch_size, H_fresnel_stack, \
                                   H_NA, Nx, low_res_obj_stack, step_size, max_internal_iter, \
                                   merit_stopping_point, loss_low_res, i, num_wavelengths)
    
                loss = tf.reduce_sum(tf.square(predicted_mat - tf.cast(trainingSet, tf.float32))) #l2 loss
                

            elif image_modality == 'phase':
    
                if max_internal_iter > 0:
                    predicted_mat = iterative_solver_phase(predicted_mat, PSF, batch_size, H_fresnel, \
                                   H_NA, Nx, low_res_obj_stack, step_size, max_internal_iter, \
                                   merit_stopping_point, loss_low_res, i)
    
                loss = calculate_loss_FP(predicted_mat, trainingSet)

            
        loss_grad = grad_diff_loss(predicted_mat, trainingSet)

    return loss, loss_grad, optical_element, low_res_obj_stack_nonoise, low_res_obj_stack, predicted_mat


def average_gradients(tower_grads, take_average=True): # Take
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, v in grad_and_vars:
#      print('*'*100)
#      print(g)
#      print(v)
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    
    if take_average:
        grad = tf.reduce_mean(grad, 0)
    else:
        grad = tf.reduce_sum(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads




def load_optical_parameters(training_data_folder,image_modality):
    optical_parameters_dict = {}
    if image_modality=='FP' or image_modality == 'FP_PP':
        optical_parameters_dict['Ns_mat'] = np.load(training_data_folder + '/Ns_mat.npy')
        optical_parameters_dict['scale_mat'] = np.load(training_data_folder + '/scale_mat.npy')
        optical_parameters_dict['cen0'] = np.load(training_data_folder + '/cen0.npy')
        optical_parameters_dict['P'] = np.load(training_data_folder + '/P.npy')
        optical_parameters_dict['H0'] = np.load(training_data_folder + '/H0.npy')
        optical_parameters_dict['Np'] = np.load(training_data_folder + '/Np.npy')
        optical_parameters_dict['numLEDs'] = np.load(training_data_folder + '/numLEDs.npy')
        optical_parameters_dict['N_obj'] = np.load(training_data_folder + '/N_obj.npy')
        optical_parameters_dict['N_patch'] = np.load(training_data_folder + '/N_patch.npy')
        optical_parameters_dict['num_patches'] = np.load(training_data_folder + '/num_patches.npy')
        optical_parameters_dict['Nx_lowres'] = np.load(training_data_folder + '/Np.npy')
        optical_parameters_dict['Nx_highres'] = np.load(training_data_folder + '/N_obj.npy')
        optical_parameters_dict['NAfilter_synthetic'] = np.load(training_data_folder + '/NAfilter_synthetic.npy')
        optical_parameters_dict['Nz'] = 1

        num_elements = optical_parameters_dict['numLEDs']

    elif image_modality=='STORM':
        optical_parameters_dict['ZernikePolyMat'] = np.load(training_data_folder + '/ZernikePolyMat.npy')
        optical_parameters_dict['H_NA'] = np.load(training_data_folder + '/H_NA.npy')
        optical_parameters_dict['H_fresnel_stack'] = np.load(training_data_folder + '/H_fresnel_stack.npy')
        optical_parameters_dict['Nx_lowres'] = np.load(training_data_folder + '/Nx.npy')
        optical_parameters_dict['Nx_highres'] = np.load(training_data_folder + '/Nx.npy')
        optical_parameters_dict['Nz'] = np.load(training_data_folder + '/Nz.npy')
        optical_parameters_dict['num_wavelengths'] = np.load(training_data_folder + '/num_wavelengths.npy')
        optical_parameters_dict['numCoeff'] = np.load(training_data_folder + '/numCoeff.npy')

        num_elements = optical_parameters_dict['numCoeff']

    elif image_modality=='HE':
        only_noise = np.load(training_data_folder + '/only_noise.npy') # Option to keep magnification/resolution the same, but add noise
        optical_parameters_dict['only_noise'] = only_noise

        optical_parameters_dict['Nx_lowres'] = np.load(training_data_folder + '/Nx_lowres.npy')
        optical_parameters_dict['Nx_highres'] = np.load(training_data_folder + '/Nx_highres.npy')
        optical_parameters_dict['num_wavelengths'] = np.load(training_data_folder + '/num_wavelengths.npy')

        if not(only_noise):
            optical_parameters_dict['wavelength'] = np.load(training_data_folder + '/wavelength.npy')
            optical_parameters_dict['h_blur'] = np.load(training_data_folder + '/h_blur.npy')
            optical_parameters_dict['high_magn'] = np.load(training_data_folder + '/high_magn.npy')
            optical_parameters_dict['low_magn'] = np.load(training_data_folder + '/low_magn.npy')
            optical_parameters_dict['dpix_c'] = np.load(training_data_folder + '/dpix_c.npy')
            optical_parameters_dict['low_NA'] = np.load(training_data_folder + '/low_NA.npy')

        num_elements = None

    elif image_modality=='phase':
        optical_parameters_dict['ZernikePolyMat'] = np.load(training_data_folder + '/ZernikePolyMat.npy')
        optical_parameters_dict['H_NA'] = np.load(training_data_folder + '/H_NA.npy')
        optical_parameters_dict['H_fresnel'] = np.load(training_data_folder + '/H_fresnel.npy')
        optical_parameters_dict['Nx_lowres'] = np.load(training_data_folder + '/Nx.npy')
        optical_parameters_dict['Nx_highres'] = np.load(training_data_folder + '/Nx.npy')
        optical_parameters_dict['numCoeff'] = np.load(training_data_folder + '/numCoeff.npy')
        optical_parameters_dict['Nz'] = 1

        num_elements = optical_parameters_dict['numCoeff']


    return optical_parameters_dict, num_elements

###### Functions for H&E

def NAfilter(L,wavelength,NA,m):

    #L is the side length of the observation and source fields (assume square fields), L is a float
    #wavelength is the free space wavelength

    dx=L/m
    k=1./wavelength #wavenumber #2*pi/wavelength #1./wavelength

    fx=tf.linspace(-1/(2*dx),1/(2*dx)-1/L,m) #freq coords

    FX,FY=tf.meshgrid(fx,fx)

    zeros = tf.cast(tf.zeros([m,m]),dtype=tf.complex64)
    ones = tf.cast(tf.ones([m,m]),dtype=tf.complex64)

    H = tf.where(tf.sqrt(FX**2+FY**2)<=NA*k,ones,zeros)


#    H=fftshift(H,m,m)

    return H

def make_even(Np):

    if np.ceil(Np)%2: # make Np even
        Np = np.floor(Np)
    else:
        Np = np.ceil(Np)

    Np = int(Np)

    if Np % 2:
        Np += 1

    return Np

def change_magn_img(img0, m, h_blur, high_magn, low_magn, dpix_c, wavelength, low_NA):

    img = tf.cast(img0, dtype=tf.complex64)
    img = img[0:m,0:m]

    L = m*dpix_c/high_magn # [m]

    cen = [m//2,m//2]
    Np = m*low_magn/high_magn

    Np = make_even(Np)

    H_NA_R = NAfilter(L,wavelength[0],low_NA,m)
    H_NA_G = NAfilter(L,wavelength[1],low_NA,m)
    H_NA_B = NAfilter(L,wavelength[2],low_NA,m)


    img_r = incoherent_filter_H(img[:,:,0], m, H_NA_R, cen, Np, h_blur)
    img_g = incoherent_filter_H(img[:,:,1], m, H_NA_G, cen, Np, h_blur)
    img_b = incoherent_filter_H(img[:,:,2], m, H_NA_B, cen, Np, h_blur)

    img_r = tf.expand_dims(img_r, axis=2)
    img_g = tf.expand_dims(img_g, axis=2)
    img_b = tf.expand_dims(img_b, axis=2)
    img=tf.concat([img_r,img_g,img_b],axis=2)

#    img = tf.stack([img_r, img_g, img_b], axis=2)

    return img0, img

def incoherent_filter_H(u1, m, H, cen, Np, h_blur):

#    H0 = H

    H = F(Ft(H,m,m)*tf.conj(Ft(H,m,m)),m,m)
    H=H/H[m//2,m//2]

    U1 = F(u1,m,m)

    U2=H*U1

    #convolve with n x n filter with n = high_magn/low_magn
    U2 = U2 * F(h_blur,m,m)

    U2 = downsamp(U2,cen,Np)

    u2=Ft(U2,Np,Np)/(m/Np)**2

    u2 = tf.real(u2)

    # Clip u2 at 0 and 255

    zeros = tf.cast(tf.zeros([Np,Np]),dtype=tf.float32)
    ones = 255*tf.cast(tf.ones([Np,Np]),dtype=tf.float32)

    u2 = tf.where(u2>255,ones,u2)
    u2 = tf.where(u2<0,zeros,u2)


#    u2 = tf.cast(u2, dtype=tf.uint8)
#    u2=Ft(U2,m,m)

    return u2

def filter_function_NA(u1,H_NA,Nx,incoherent=1):
    #u1 is the source plane field
    #Nx is u1.shape[0]
    #dx is L/m

    H = H_NA

    if incoherent:
        H=tf.fft2d(tf.ifft2d(H)*tf.conj(tf.ifft2d(H)))
        #H=H/H[0,0]

        U1=fftshift(u1,Nx,Nx)
        U1=tf.fft2d(U1)

        U2=H*U1
        u2=fftshift(tf.ifft2d(U2),Nx,Nx)

    else:
        U1=fftshift(u1,Nx,Nx)
        U1=tf.fft2d(U1)

        U2=H*U1
        u2=fftshift(tf.ifft2d(U2),Nx,Nx)
        
#        u2 = u2*np.conj(u2) # make into intensity object

    return u2


def read_img_file(img_path, channels):
    img_file = tf.read_file(img_path)
    img0 = tf.image.decode_image(img_file, channels=channels)
    return img0

def change_magn(img_path, m, h_blur, high_magn, low_magn, dpix_c, wavelength, low_NA, num_wavelengths):

    img0 = read_img_file(img_path, num_wavelengths)

    return change_magn_img(img0, m, h_blur, high_magn, low_magn, dpix_c, wavelength, low_NA)


def change_magn_batch(img_stack, m, h_blur, high_magn, low_magn, dpix_c, wavelength, low_NA, batch_size):

    for ii in range(batch_size):


        ( _ , img) =change_magn_img(img_stack[ii,:,:,:], m, h_blur, high_magn, low_magn, dpix_c, wavelength, low_NA)

        img = tf.expand_dims(img,axis=0)

        if ii == 0:
            img_stack_new = img
        else:
            img_stack_new = tf.concat([img_stack_new,img],0)

#    img_stack_new = tf.cast(img_stack_new, dtype=tf.complex64)

    return img_stack_new

def add_poisson_noise(img0, img, noise_multiplier):

    img = noise_multiplier*tf.cast(img, dtype=tf.float32)
    img = tf.random_poisson(img, [1])
    img = img/noise_multiplier

    # Clip img at 0 and 255

    zeros = tf.cast(tf.zeros_like(img),dtype=tf.float32)
    ones = 255*tf.cast(tf.ones_like(img),dtype=tf.float32)

    img = tf.where(img>255,ones,img)
    img = tf.where(img<0,zeros,img)

    img = tf.cast(img,dtype=tf.uint8)

    img = tf.squeeze(img, axis=0)

    return img0, img


def upsample_rgb(low_res_obj_rgb, Np, N_obj, num_wavelengths):

    for c in range(num_wavelengths):
        upsampled_obj = upsample(low_res_obj_rgb[:,:,c], Np, N_obj)

        upsampled_obj = tf.expand_dims(upsampled_obj, axis=2)

        if c == 0:
            upsampled_obj_stack = upsampled_obj
        else:
            upsampled_obj_stack = tf.concat([upsampled_obj_stack,upsampled_obj],2)

    return upsampled_obj_stack


def make_iterator_numpy(training_dataset, batch_size, num_GPUs, shuffle=False):
    tr_data = tf.data.Dataset.from_tensor_slices(training_dataset)
    tr_indices =tf.data.Dataset.from_tensor_slices(tf.constant(np.arange(0,training_dataset.shape[0],1),dtype = tf.int32))

    tr_data = tf.data.Dataset.zip((tr_data, tr_indices))

    tr_data = tr_data.repeat()

    if shuffle:
        tr_data = tr_data.shuffle(training_dataset.shape[0])

    tr_data = tr_data.batch(batch_size*num_GPUs)
    
    return tr_data

def make_iterator_FP_PP(training_dataset, lowres_training_dataset, \
                        batch_size, num_GPUs, shuffle=False):   

    tr_data = tf.data.Dataset.from_tensor_slices(training_dataset)
    tr_data_lowres = tf.data.Dataset.from_tensor_slices(lowres_training_dataset)
    tr_indices =tf.data.Dataset.from_tensor_slices(tf.constant(np.arange(0,training_dataset.shape[0],1),dtype = tf.int32))

    tr_data = tf.data.Dataset.zip((tr_data, tr_data_lowres, tr_indices))

    tr_data = tr_data.repeat()

    if shuffle:
        tr_data = tr_data.shuffle(training_dataset.shape[0])

    tr_data = tr_data.batch(batch_size*num_GPUs)
    
    return tr_data

def make_complex_img(img_path, Nx, phase = True, intensity = False, library = tf):
    if library == tf:
        img0 = read_img_file(img_path, None)
        img0 = library.reduce_sum(img0,axis=2)
        img0 = img0[Nx/2:-Nx/2,Nx/2:-Nx/2]
        img0 = library.cast(img0,dtype=library.complex64)
    else:
        files = glob.glob(img_path)
        fileI = files[0]
        img0 = misc.imread(fileI)
        img0 = library.sum(img0, axis=2)
        img0 = img0[0:Nx,0:Nx]
        img0 = img0.astype(library.complex64)
        
    img0 = library.exp(1j*math.pi*1.0*(255.0*3-img0)/(255.0*3))
    return img0

def make_iterator_filename_FP(training_dataset, batch_size, num_GPUs, optical_parameters_dict, \
                              shuffle=False):
    
    Nx = optical_parameters_dict['Nx_highres']
    H_NA = optical_parameters_dict['NAfilter_synthetic']
    
    training_dataset = tf.constant(training_dataset.tolist())
    tr_data = tf.data.Dataset.from_tensor_slices(training_dataset)
    tr_indices =tf.data.Dataset.from_tensor_slices(tf.constant(np.arange(0,int(training_dataset.shape[0]),1),dtype = tf.int32))

    filter_function_NA_lambda = lambda u1: filter_function_NA(u1,H_NA,Nx,incoherent=0)
    make_complex_img_lambda = lambda img_path: make_complex_img(img_path,Nx)

    tr_data = tr_data.map(make_complex_img_lambda)
    tr_data = tr_data.map(filter_function_NA_lambda)
    tr_data = tf.data.Dataset.zip((tr_data, tr_indices))

    tr_data = tr_data.repeat()

    if shuffle:
        tr_data = tr_data.shuffle(training_dataset.shape[0])

    tr_data = tr_data.batch(batch_size*num_GPUs)
    
    return tr_data




def create_STORM_stack(img_path, Nx):
    img0 = read_img_file(img_path, None)
    img0 = img0[0:Nx,0:Nx,:]
    img0 = tf.expand_dims(img0, axis = 2)
    img0 = tf.cast(img0, tf.float32)
    return img0
    
    
    
def make_iterator_filename_STORM(training_dataset, batch_size, num_GPUs, optical_parameters_dict, \
                                 shuffle=False):
    
    Nx = optical_parameters_dict['Nx_highres']
    
    training_dataset = tf.constant(training_dataset.tolist())
    tr_data = tf.data.Dataset.from_tensor_slices(training_dataset)
    tr_indices =tf.data.Dataset.from_tensor_slices(tf.constant(np.arange(0,int(training_dataset.shape[0]),1),dtype = tf.int32))


    create_STORM_stack_lambda = lambda img_path: create_STORM_stack(img_path, Nx)
    
    tr_data = tr_data.map(create_STORM_stack_lambda)
    tr_data = tf.data.Dataset.zip((tr_data, tr_indices))

    tr_data = tr_data.repeat()

    if shuffle:
        tr_data = tr_data.shuffle(training_dataset.shape[0])

    tr_data = tr_data.batch(batch_size*num_GPUs)
    
    return tr_data

def make_iterator_filename(training_dataset, add_noise, poisson_noise_multiplier, batch_size, \
                           num_GPUs, optical_parameters_dict, shuffle = False):

    only_noise = optical_parameters_dict['only_noise']
    num_wavelengths = optical_parameters_dict['num_wavelengths']

    training_dataset = tf.constant(training_dataset.tolist())
    tr_data = tf.data.Dataset.from_tensor_slices(training_dataset)
    tr_indices =tf.data.Dataset.from_tensor_slices(tf.constant(np.arange(0,int(training_dataset.shape[0]),1),dtype = tf.int32))



    if only_noise:
        read_img_file_lambda = lambda img_path: read_img_file(img_path, num_wavelengths)
        tr_data = tr_data.map(lambda img_path: (read_img_file_lambda(img_path),read_img_file_lambda(img_path)))
    else:
        change_magn_lambda = lambda img_path: change_magn(img_path,
                                                  optical_parameters_dict['Nx_highres'],
                                                  optical_parameters_dict['h_blur'],
                                                  optical_parameters_dict['high_magn'],
                                                  optical_parameters_dict['low_magn'],
                                                  optical_parameters_dict['dpix_c'],
                                                  optical_parameters_dict['wavelength'],
                                                  optical_parameters_dict['low_NA'],
                                                  num_wavelengths)
        tr_data = tr_data.map(change_magn_lambda)

    add_poisson_noise_lambda = lambda img0, img: add_poisson_noise(img0, img, poisson_noise_multiplier)

    if only_noise and not(add_noise):
        print('Warning: only_noise = True and add_noise = False!!')

    if add_noise:
        tr_data = tr_data.map(add_poisson_noise_lambda)

    tr_data = tf.data.Dataset.zip((tr_data, tr_indices))

    tr_data = tr_data.repeat()
    
    if shuffle:
        tr_data = tr_data.shuffle(int(training_dataset.shape[0]))    
    
    tr_data = tr_data.batch(batch_size*num_GPUs)

    return tr_data




def tower_loss_HE(low_res_obj_batch, \
                  high_res_obj_batch, training, \
                  use_batch_norm, variance_reg, sqrt_reg, \
                  dropout_prob, layers_dropout, batch_size, max_internal_iter, merit_stopping_point, \
                  optical_parameters_dict, autoencode_init_type, init_type_bias, init_type_resid, \
                  kernel_multiplier, num_layers_autoencode, skip_interval, num_blocks):
    
    only_noise = optical_parameters_dict['only_noise']
    Nx_lowres = optical_parameters_dict['Nx_lowres']
    Nx_highres = optical_parameters_dict['Nx_highres']
    num_wavelengths = optical_parameters_dict['num_wavelengths']

    if not(only_noise):
        wavelength = optical_parameters_dict['wavelength']
        h_blur = optical_parameters_dict['h_blur']
        high_magn = optical_parameters_dict['high_magn']
        low_magn = optical_parameters_dict['low_magn']
        dpix_c = optical_parameters_dict['dpix_c']
        low_NA = optical_parameters_dict['low_NA']


    with tf.variable_scope("optical_transform"):

        if only_noise:
            upsampled_obj_stack = low_res_obj_batch
        else:
            # upsample the low_res_obj_stack
            simple_upsample = 0
            if simple_upsample:
                low_res_obj_batch = tf.cast(low_res_obj_batch, tf.complex64)
                for ii in range(batch_size):
                    upsampled_obj = upsample_rgb(low_res_obj_batch[ii,:,:,:], Nx_lowres, Nx_highres, num_wavelengths)

                    upsampled_obj = tf.expand_dims(upsampled_obj,axis=0)
                    if ii == 0:
                        upsampled_obj_stack = upsampled_obj
                    else:
                        upsampled_obj_stack = tf.concat([upsampled_obj_stack,upsampled_obj],0)
            else:
                upsampled_obj_stack = tf.image.resize_images(low_res_obj_batch, [int(Nx_highres), int(Nx_highres)], \
                                                       method=tf.image.ResizeMethod.BICUBIC, align_corners=False)

        upsampled_obj_stack = tf.cast(upsampled_obj_stack, tf.float32)

    with tf.variable_scope("neural_net"):

        num_nets = num_wavelengths
        input_layer_list = []

        for w in range(num_wavelengths):

            input_layer = tf.expand_dims(upsampled_obj_stack[:,:,:,w], axis=3)
            input_layer_list.append(input_layer)

#        len(input_layer) should equal num_nets

#        input_layer_r = tf.expand_dims(upsampled_obj_stack[:,:,:,0], axis=3)
#        input_layer_g = tf.expand_dims(upsampled_obj_stack[:,:,:,1], axis=3)
#        input_layer_b = tf.expand_dims(upsampled_obj_stack[:,:,:,2], axis=3)
#
#
#        input_layer_list = [input_layer_r, input_layer_g, input_layer_b]

        predicted_mat = find_predicted_mat('HE', num_nets, input_layer_list, Nx_highres, batch_size, layers_dropout, dropout_prob, 
                       use_batch_norm, autoencode_init_type, init_type_bias, init_type_resid, kernel_multiplier, variance_reg, training, \
                       num_layers_autoencode, skip_interval, num_blocks, optical_parameters_dict)
        
    with tf.variable_scope("neural_net"): 
            

#        ### convert predicted_mat down to low_res_obj_predicted
        if only_noise:
            low_res_obj_batch_predicted = low_res_obj_batch

        else:
            low_res_obj_batch_predicted = change_magn_batch(predicted_mat, Nx_highres, h_blur, \
                                                            high_magn, low_magn, dpix_c, wavelength, low_NA, batch_size)

            low_res_obj_batch_predicted = tf.cast(low_res_obj_batch_predicted, tf.float32)
            low_res_obj_batch = tf.cast(low_res_obj_batch, tf.float32)
            loss_low_res = tf.reduce_sum(tf.square(low_res_obj_batch - low_res_obj_batch_predicted))


###################
        step_size = variable_on_cpu('step_size', 1e-7, tf.float32, trainable = False) # XXX make step_size trainable

    with tf.variable_scope("iteration_variable"):
        i = tf.Variable(0, dtype=tf.int32, trainable=False)

    with tf.variable_scope("neural_net"):

        if not(only_noise):
            if max_internal_iter > 0:
                predicted_mat, loss_low_res = iterative_solver_HE(predicted_mat, Nx_highres, h_blur, \
                               high_magn, low_magn, dpix_c, wavelength, \
                               low_NA, low_res_obj_batch, step_size, max_internal_iter, \
                               merit_stopping_point, loss_low_res, i, \
                               batch_size)

        high_res_obj_batch = tf.cast(high_res_obj_batch, tf.float32)

        loss = tf.reduce_sum(tf.square(high_res_obj_batch - predicted_mat))

        if only_noise:
            loss_low_res = loss
        
        loss_grad = grad_diff_loss(predicted_mat, high_res_obj_batch)

    return loss, loss_grad, loss_low_res, low_res_obj_batch_predicted, low_res_obj_batch, predicted_mat
    #output names are: loss, loss_grad, optical_element, low_res_obj_stack_nonoise, low_res_obj_stack, predicted_mat
    #for only_noise = True, loss_low_res is set to loss, and low_res_obj_batch_predicted is set to low_res_obj_batch
