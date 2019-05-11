import tensorflow as tf
import numpy as np


def variable_on_cpu(name, initial_value, dtype, trainable=True, constraint=None, shape=None):
    """Helper to create a Variable stored on CPU memory."""

    with tf.device('/cpu:0'):
        if shape == None:
            var = tf.get_variable(name, dtype=dtype, initializer=initial_value, trainable=trainable, constraint=constraint)
        else:
            var = tf.get_variable(name, dtype=dtype, initializer=initial_value, trainable=trainable, constraint=constraint, shape=shape)
    return var

#def convLayer(layer_input, sizes, kernel_init, padding="same", activation=tf.nn.relu):
#    """Creates a convolutional layer.
#    Sizes is list with [filters, kernel_x, kernel_y]"""
#
#    conv = tf.layers.conv2d(
#        inputs=layer_input,
#        filters=sizes[0],
#        kernel_size=sizes[1:],
#        padding=padding,
#        activation=activation,
#        kernel_initializer=kernel_init
#    )
#
#    return conv

def conv_layer(x_in, w, strides=[1,1,1,1], padding="SAME", name="Conv_", name_index=0, init_type="trunc_norm"):
    b = variable_on_cpu('bias', getConvInitializer(1, 1, init_type=init_type), tf.float32)
    return tf.nn.conv2d(x_in, w, strides=strides, padding=padding, name=name+"-"+str(name_index)) + b

def deconv_layer(x_in, w, obj_size, batch_size, strides=[1,1,1,1], padding="SAME", name="DeConv_", name_index=0, num_channels=1, init_type="trunc_norm"):
    b = variable_on_cpu('bias', getConvInitializer(1, 1, init_type=init_type), tf.float32)
    return tf.nn.conv2d_transpose(x_in, w, [batch_size, obj_size, obj_size, num_channels], strides, padding=padding, name=name+"-"+str(name_index)) + b

#def conv_relu(x_in, w, strides=[1,1,1,1], padding="SAME", name="Conv_", name_index=0):
#    return tf.nn.relu(tf.nn.conv2d(x_in, w, strides=strides, padding=padding), name=name+"-"+str(name_index))

def maxout(conv_layer1, conv_layer2):
    return tf.maximum(conv_layer1, conv_layer2)
#    return tf.where(tf.abs(conv_layer1)>tf.abs(conv_layer2),conv_layer1,conv_layer2)
#    return tf.minimum(conv_layer1, conv_layer2)

#def conv_maxout(x_in, w, strides=[1,1,1,1], padding="SAME", name="Conv_Maxout", num_units=1, name_index=0):
#    return tf.contrib.layers.maxout(tf.nn.conv2d(x_in, w, strides=strides, padding=padding), num_units, name=name+"-"+str(name_index))

#def deconv_relu(x_in, w, obj_size, batch_size, strides=[1,1,1,1], padding="SAME", name="DeConv_ReLU", name_index=0, num_channels=1):
#    print('w: ', w)
#    return tf.nn.relu(tf.nn.conv2d_transpose(x_in, w, [batch_size, obj_size, obj_size, num_channels], strides, padding=padding), name=name+"-"+str(name_index))

#def deconv_maxout(x_in, w, obj_size, batch_size, strides=[1,1,1,1], padding="SAME", name="DeConv_Maxout", num_units=1, name_index=0, num_channels=1):
#    return tf.contrib.layers.maxout(tf.nn.conv2d_transpose(x_in, w, [batch_size, obj_size, obj_size, num_channels], strides, padding=padding), num_units, name=name+"-"+str(name_index))

#def deconv_maxout_upsample(x_in, w, obj_size, strides=[1,1,1,1], padding="SAME", size=1, name="DeConv_Maxout_Upsample", name_index=0):
#    upsampled = tf.image.resize_images(x_in, size=(obj_size, size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#    return tf.contrib.layers.maxout(tf.nn.conv2d(upsampled, w, strides=strides, padding=padding), 1, name=name+"-"+str(name_index))

def getConvInitializer(width, height, num_channels=1, filter_depth=1, init_type="trunc_norm"):
    """ """

    if init_type == "delta":
        init = np.zeros((width, height, num_channels, filter_depth),dtype=np.float32)
        init[width/2][height/2][0][0] = 1
        return init

    if init_type == "negative_delta":
        init = np.zeros((width, height, num_channels, filter_depth),dtype=np.float32)
        init[width/2][height/2][0][0] = -10000
        return init

    elif init_type == "zeros":
        init = np.zeros((width, height, num_channels, filter_depth),dtype=np.float32)
        return init
       
    elif init_type == "big_delta":
        inner_size = (10, 10)
        pad_size = ((width-inner_size[0])/2, (height-inner_size[1])/2)

        mat = tf.constant(np.ones([inner_size[0], inner_size[1]]), dtype=tf.float32)

        if width % 2 == 0:
            paddings = tf.constant([[pad_size[1], pad_size[1]], [pad_size[0], pad_size[0]]])
        else:
            paddings = tf.constant([[pad_size[1], pad_size[1]+1], [pad_size[0], pad_size[0]+1]])

        p = tf.pad(mat, paddings, "CONSTANT")

        return tf.reshape(p, (width, height, num_channels, filter_depth))

    elif init_type == "random":
        return tf.cast(np.random.rand(width, height, num_channels, filter_depth)-0.5, tf.float32)

    elif init_type == "trunc_norm":
        return tf.truncated_normal([width, height, num_channels, filter_depth], stddev=0.1)

    elif init_type == "gauss":
        sigma = 0.5
        halfwidth_x = np.arange(-width/2, width/2)
        halfwidth_y = np.arange(-height/2, height/2)
        xx, yy = np.meshgrid(halfwidth_x, halfwidth_y)
        gauss = np.reshape(np.exp(-1/float((2*sigma**2)) * (xx**2+yy**2)), (width, height, num_channels, filter_depth))
        return tf.cast(gauss, tf.float32)

    elif init_type == "rand_unif":
        return tf.random_uniform([width, height, num_channels, filter_depth])

    elif init_type == "ones":
        return tf.ones([width, height, num_channels, filter_depth])

    elif init_type == "fill":
        fill_value = 100.
        filled = tf.fill([width, height], fill_value)
        return tf.reshape(filled, (width, height, num_channels, filter_depth))

    else:
        raise ValueError("Please provide a valid init_type argument.")



def residualLayer(input1, skipped_input, name="Resid_ReLU", name_index=0, init_type="trunc_norm"): #activation=tf.nn.relu
    """Returns residual layer that adds inputs and passes them through ReLU"""

    resid_multiplier = variable_on_cpu('resid_multiplier', getConvInitializer(1, 1, init_type=init_type), tf.float32)
    added = tf.add(resid_multiplier*input1, skipped_input)
    
#    return activation(added, name=name+"-"+str(name_index))
    
    return added

def dropoutLayer(input_, rate=0.7, training=True, name="Dropout"):
    """ Wrapper for dropout layer """

    return tf.layers.dropout(input_, rate=rate, training=training, name=name)

def batch_norm_layer(conv_layer, variance_reg):
    """ Wrapper for batch norm layer """

    # Calculate batch mean and variance
    batch_mean, batch_var = tf.nn.moments(conv_layer,[0,1,2,3])

    # Apply the initial batch normalizing transform
    conv_layer = (conv_layer - batch_mean) / tf.sqrt(batch_var + tf.constant(variance_reg,tf.float32))

    # Create two new parameters, scale and beta (shift)
    scale = variable_on_cpu('scale', 1.0, tf.float32)
    beta = variable_on_cpu('beta', 0.0, tf.float32)

    # Scale and shift to obtain the final output of the batch normalization

    conv_layer = scale * conv_layer + beta

    return conv_layer

def create_conv_layer(kernel_length, input_layer, input_channels, output_channels):
    # input_channels = int(input_layer.shape[3])
    # output_channels = Nz
    kernel = variable_on_cpu('kernel', tf.truncated_normal([kernel_length,kernel_length,input_channels,output_channels], \
                                                            mean=0, stddev=0.1, dtype=tf.float32), tf.float32)
    biases = variable_on_cpu('biases', tf.truncated_normal([output_channels], mean=0, stddev=0.1, dtype=tf.float32), tf.float32)
    conv_layer = tf.nn.conv2d(input_layer, kernel, [1, 1, 1, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, biases)
    return conv_layer

def create_residual_layer_maxout(kernel_length_vec,input_layer,use_batch_norm,input_channels,output_channels,variance_reg,dropout_prob,training):
    # kernel_length_vec contains 2 lengths

    input_layer = tf.layers.dropout(inputs=input_layer, rate=dropout_prob, training=training) #rate is percentage that is dropped

    with tf.variable_scope("conv1"):
        conv1 = create_conv_layer(kernel_length_vec[0],input_layer,input_channels,output_channels)

        if use_batch_norm:
            conv1 = batch_norm_layer(conv1,variance_reg)

    with tf.variable_scope("conv2"):
        conv2 = create_conv_layer(kernel_length_vec[0],input_layer,input_channels,output_channels)
        if use_batch_norm:
            conv2 = batch_norm_layer(conv2,variance_reg)

        maxout1 = tf.maximum(conv1,conv2)

        maxout1 = tf.layers.dropout(inputs=maxout1, rate=dropout_prob, training=training) #rate is percentage that is dropped
        
    with tf.variable_scope("conv3"):
        conv3 = create_conv_layer(kernel_length_vec[1],maxout1,output_channels,output_channels)
        added_layer1 = conv3 + input_layer
        
        if use_batch_norm:
            added_layer1 = batch_norm_layer(added_layer1,variance_reg)
            
    with tf.variable_scope("conv4"):
        conv4 = create_conv_layer(kernel_length_vec[1],maxout1,output_channels,output_channels)
        added_layer2 = conv4 + input_layer
        
        if use_batch_norm:
            added_layer2 = batch_norm_layer(added_layer2,variance_reg)
            
        res_layer = tf.maximum(added_layer1,added_layer2)

    return res_layer