#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:49:13 2018

@author: vganapa1
"""

import tensorflow as tf
import numpy as np
from E30_Tensorflow_Wavelength_Functions import H_fresnel_prop, conv_layer, \
                                                dense_layer

Nx = 2**5
Ny = 2**5
dx = 200e-9 #[m]
dy = dx
z = 3e-6
num_iter = 1000
kernel_length = 32
num_conv_layers = 5
#restore_model = False
learning_rate = 0.01

single_pt = np.zeros([Nx,Ny])
single_pt[Nx//2,Ny//2]=1


with tf.Graph().as_default():
    

    wavelength = tf.placeholder(tf.float32)
    single_pt_0 = tf.constant(single_pt,dtype=tf.complex64)
    

    img_2D = H_fresnel_prop(single_pt_0,Nx,Ny,dx,dy,z,wavelength)
                      
    img_2D = tf.expand_dims(img_2D, axis=2)
    img_2D = tf.expand_dims(img_2D, axis=0)

    input_layer = img_2D
    
    for i in range(num_conv_layers):
        with tf.variable_scope('conv_layer_' + str(i)):
            output_layer = conv_layer(kernel_length, input_layer, 1, 1)
            input_layer = output_layer
    
    input_layer = dense_layer(input_layer,Nx,Ny,1,name='dense_layer')
    input_layer = tf.squeeze(input_layer, axis=0)

    wavelength_guess = input_layer[0]
    MSE = (wavelength-wavelength_guess)**2 
    

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(MSE)
    
    init_op = tf.global_variables_initializer()
#    saver = tf.train.Saver() 
        
    with tf.Session() as sess: 
        sess.run(init_op)
#        
#        if restore_model:
#            saver.restore(sess, 'model.ckpt')

        for i in range(num_iter):
            wavelength_0 = np.random.randint(450,750)*1e-9
#            wavelength_0 = 500*1e-9

            [_,MSE_0,wavelength_1,img_2D_0] = sess.run([train,MSE,wavelength_guess, img_2D], \
                                                     feed_dict = {wavelength: wavelength_0})

            print('MSE: ' + str(MSE_0))
            print(wavelength_0)
            print(wavelength_1)
            print
   
#        save_path = saver.save(sess, os.getcwd() + '/model.ckpt')
#        print("Model saved in file: %s" % save_path) 
        
#        plt.figure()
#        plt.imshow(PSF_0)
#        plt.colorbar()
        