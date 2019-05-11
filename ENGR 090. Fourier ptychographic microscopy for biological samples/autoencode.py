import tensorflow as tf
import layer_defs as ld
from layer_defs import variable_on_cpu


try:
    import graphviz as gv
    has_graphviz = True
except ImportError:
    print('System does not have graphviz.')
    has_graphviz = False

class StackedAutoEncoder(object):

    def __init__(self, x_input, cnn_depth, obj_size, batch_size, 
                 training, kernel_multiplier, num_blocks, 
                 skip_interval=2, conv_activation='maxout', 
                 deconv_activation='maxout', resid=ld.residualLayer, 
                 dropout_count=0, dropout_prob=0.2, 
                 create_graph_viz=True, decay_lr_flag=False,
                 variance_reg=1e-8, use_batch_norm=False, 
                 init_type='random', init_type_bias='trunc_norm', 
                 init_type_resid='ones'):

        self.all_blocks = []
        curr_input = x_input
        for i in range(num_blocks):
            with tf.variable_scope('net' + str(i)):
                ae = AutoEncoder(curr_input, cnn_depth, obj_size, 
                                 batch_size, training, 
                                 kernel_multiplier, skip_interval=2, 
                                 conv_activation='maxout', 
                                 deconv_activation='maxout',
                                 resid=ld.residualLayer, 
                                 dropout_count=0, dropout_prob=0.2, 
                                 create_graph_viz=True, 
                                 decay_lr_flag=False,
                                 variance_reg=1e-8, 
                                 use_batch_norm=False, 
                                 init_type='random', 
                                 init_type_bias='trunc_norm', 
                                 init_type_resid='ones')
                curr_input = ae.get_prediction()
                print('curr_input')
                print(curr_input)
                self.all_blocks.append(ae)

        print(self.all_blocks)

    def get_stacked_prediction(self):
        print('self.all_blocks[-1].get_prediction()')
        print(self.all_blocks[-1].get_prediction())
        
        return self.all_blocks[-1].get_prediction()


class AutoEncoder(object):
    def __init__(self, x_input, cnn_depth, obj_size, batch_size, training, kernel_multiplier, skip_interval=2, conv_activation='maxout', deconv_activation='maxout',
                    resid=ld.residualLayer, dropout_count=0, dropout_prob=0.2, create_graph_viz=True, decay_lr_flag=False,
                    variance_reg=1e-8, use_batch_norm=False, init_type='random', init_type_bias='trunc_norm', init_type_resid='ones'):

        """Constructor for AutoEncoder class"""

        self.batch_size = batch_size                                        # batch size
        self.obj_size = obj_size                                            # size of output object
        self.x_input = x_input                                              # x_input tensor
        self.cnn_depth = cnn_depth                                          # number of CNN layers
        self.skip_interval = skip_interval                                  # interval for resid (must have cnn_depth % skip_interval == 0)
        self.num_resid = self.cnn_depth / self.skip_interval                # number of residual layers
        self.conv_activation = conv_activation                              # conv activation/layer def
        self.deconv_activation = deconv_activation                          # deconv activation/layer def
        self.resid = resid                                                  # residual layer def
        self.encode_layers, self.decode_layers = [self.x_input], []         # lists for encode and decode layers
        self.kernel_multiplier = kernel_multiplier                          # kernel multiplier (kernel gets smaller as we get closer to center of network)
        self.dropout_count = dropout_count                                  # number of dropout layers
        self.dropout_prob = dropout_prob                                    # dropout probability
        self.create_graph_viz = create_graph_viz
        self.kernel_sizes = range(self.kernel_multiplier, (self.cnn_depth + 1) * self.kernel_multiplier, self.kernel_multiplier)
        self.variance_reg = variance_reg
        self.use_batch_norm = use_batch_norm
        self.init_type = init_type
        self.init_type_bias = init_type_bias
        self.init_type_resid = init_type_resid

        if has_graphviz:
            g = gv.Digraph(format='svg')

        dropout_encode = []

        for i in range(self.cnn_depth):

            with tf.variable_scope('conv' + str(i)):

                kernel_index = self.cnn_depth-i-1
                # add convolutional layer
                initializer = ld.getConvInitializer(self.kernel_sizes[kernel_index], self.kernel_sizes[kernel_index], init_type=self.init_type)

                # add dropout layer
                if i >= self.cnn_depth - self.dropout_count:
                    dropout = ld.dropoutLayer(self.encode_layers[-1], rate=self.dropout_prob, training=training)

                    if has_graphviz:
                        self.__add_node_and_edge(g, self.encode_layers[-1], dropout)

                    self.encode_layers.append(dropout)

                    dropout_encode.append(i)

                if self.conv_activation == 'relu':
                    cnn_layer = ld.conv_layer(self.encode_layers[-1],
                                              variable_on_cpu('conv_kernel' + str(i), initializer, tf.float32),
                                              name_index=i,
                                              init_type=self.init_type_bias)

                    if self.use_batch_norm:
                        cnn_layer = ld.batch_norm_layer(cnn_layer, self.variance_reg)

                    cnn_layer = tf.nn.relu(cnn_layer)


                elif self.conv_activation == 'maxout':

                    with tf.variable_scope('maxout_layer1'):

                        cnn_layer1 = ld.conv_layer(self.encode_layers[-1],
                                                   variable_on_cpu('conv_kernel1' + str(i), initializer, tf.float32),
                                                   name_index=i,
                                                   init_type=self.init_type_bias)

                        if self.use_batch_norm:
                            cnn_layer1 = ld.batch_norm_layer(cnn_layer1, self.variance_reg)

                    with tf.variable_scope('maxout_layer2'):

                        cnn_layer2 = ld.conv_layer(self.encode_layers[-1],
                                                   variable_on_cpu('conv_kernel2' + str(i) ,initializer, tf.float32),
                                                   name_index=i,
                                                   init_type=self.init_type_bias)

                        if self.use_batch_norm:
                            cnn_layer2 = ld.batch_norm_layer(cnn_layer2, self.variance_reg)

                    cnn_layer = ld.maxout(cnn_layer1, cnn_layer2)

                if has_graphviz:
                    self.__add_node_and_edge(g, self.encode_layers[-1], cnn_layer)

                self.encode_layers.append(cnn_layer)



        with tf.variable_scope('deconv0'):
            # create first deconv layer joined to last conv layer
            deconv_initializer = ld.getConvInitializer(self.kernel_sizes[0], self.kernel_sizes[0], init_type=self.init_type)

            if self.deconv_activation == 'relu':

                first_deconv = ld.deconv_layer(self.encode_layers[-1],
                                               variable_on_cpu('deconv_kernel0',deconv_initializer, tf.float32),
                                               self.obj_size, self.batch_size, name_index=0,
                                               init_type=self.init_type_bias)


                if self.use_batch_norm:
                    first_deconv = ld.batch_norm_layer(first_deconv, self.variance_reg)

                first_deconv = tf.nn.relu(first_deconv)

            elif self.deconv_activation == 'maxout':
                with tf.variable_scope('maxout_layer1'):
                    first_deconv1 = ld.deconv_layer(self.encode_layers[-1],
                                                    variable_on_cpu('deconv_kernel0_1',deconv_initializer, tf.float32),
                                                    self.obj_size, self.batch_size, name_index=0,
                                                    init_type=self.init_type_bias)

                    if self.use_batch_norm:
                        first_deconv1 = ld.batch_norm_layer(first_deconv1, self.variance_reg)

                with tf.variable_scope('maxout_layer2'):
                    first_deconv2 = ld.deconv_layer(self.encode_layers[-1],
                                                    variable_on_cpu('deconv_kernel0_2',deconv_initializer, tf.float32),
                                                    self.obj_size, self.batch_size, name_index=0,
                                                    init_type=self.init_type_bias)

                    if self.use_batch_norm:
                        first_deconv2 = ld.batch_norm_layer(first_deconv2, self.variance_reg)


                first_deconv = ld.maxout(first_deconv1, first_deconv2)


            if has_graphviz:
                self.__add_node_and_edge(g, self.encode_layers[-1], first_deconv)

            self.decode_layers.append(first_deconv)


        num_resid = 0
        num_dropout_decode = 0

        # loop to create deconvolutional and residual layers
        # possible layer types: deconvolutional, residual, dropout

        for i in range(1,self.cnn_depth):

            with tf.variable_scope('deconv' + str(i)):
                # add convolutional layer
                deconv_initializer = ld.getConvInitializer(self.kernel_sizes[i], self.kernel_sizes[i], init_type=self.init_type)

                # add dropout layer

                if i >= self.cnn_depth - self.dropout_count:
                    dropout = ld.dropoutLayer(self.decode_layers[-1], rate=self.dropout_prob, training=training)

                    if has_graphviz:
                        self.__add_node_and_edge(g, self.decode_layers[-1], dropout)

                    self.decode_layers.append(dropout)
                    num_dropout_decode += 1


                if self.deconv_activation == 'relu':
                    deconv_layer = ld.deconv_layer(self.decode_layers[-1],
                                                   variable_on_cpu('deconv_kernel' + str(i),deconv_initializer, tf.float32),
                                                   self.obj_size, self.batch_size, name_index=i,
                                                   init_type=self.init_type_bias)

                    if self.use_batch_norm:
                        deconv_layer = ld.batch_norm_layer(deconv_layer, self.variance_reg)

                    deconv_layer = tf.nn.relu(deconv_layer)

                elif self.deconv_activation == 'maxout':
                    with tf.variable_scope('maxout_layer1'):
                        deconv_layer1 = ld.deconv_layer(self.decode_layers[-1],
                                                        variable_on_cpu('deconv_kernel1' + str(i),deconv_initializer, tf.float32),
                                                        self.obj_size, self.batch_size, name_index=i,
                                                        init_type=self.init_type_bias)

                        if self.use_batch_norm:
                            deconv_layer1 = ld.batch_norm_layer(deconv_layer1, self.variance_reg)

                    with tf.variable_scope('maxout_layer2'):
                        deconv_layer2 = ld.deconv_layer(self.decode_layers[-1],
                                                        variable_on_cpu('deconv_kernel2' + str(i),deconv_initializer, tf.float32),
                                                        self.obj_size, self.batch_size, name_index=i,
                                                        init_type=self.init_type_bias)

                        if self.use_batch_norm:
                            deconv_layer2 = ld.batch_norm_layer(deconv_layer2, self.variance_reg)

                    deconv_layer = ld.maxout(deconv_layer1, deconv_layer2)

                if has_graphviz:
                    self.__add_node_and_edge(g, self.decode_layers[-1], deconv_layer)

                self.decode_layers.append(deconv_layer)

                # add residual layer if we reach a skip interval
                if (len(self.decode_layers) - num_resid - num_dropout_decode) % self.skip_interval == 0 \
                    and (len(self.decode_layers) - num_resid - num_dropout_decode) != 0 :

                    layer_i = self.cnn_depth - i - 1
                    num_dropout_encode = 0
                    for m in range(len(dropout_encode)):
                        if dropout_encode[m] < layer_i:
                            num_dropout_encode += 1


                    resid = self.resid(self.encode_layers[self.cnn_depth - i - 1 + num_dropout_encode], deconv_layer, \
                                       name_index=num_resid, init_type=self.init_type_resid)
#                    resid = self.resid(self.encode_layers[self.cnn_depth - i - 1 + num_dropout_encode], deconv_layer, name_index=num_resid)


                    if has_graphviz:
                        self.__add_node_and_edge(g, self.encode_layers[self.cnn_depth-i-1+num_dropout_encode], resid)
                        self.__add_node_and_edge(g, deconv_layer, resid)

                    self.decode_layers.append(resid)
                    num_resid += 1



        self.all_layers = self.encode_layers + self.decode_layers

        if has_graphviz:
            if self.create_graph_viz:
                g.render('img/g2')

    def __str__(self):
        return "Autoencoder-Network-" + self.x_input.name()

    def __format_graph_name(self, t):
        return t.name[:len(t.name)-2]

    def __format_graph_edge(self, t1, t2):
        return self.__format_graph_name(t1), self.__format_graph_name(t2)

    def __add_node_and_edge(self, g, t1, t2):
        new = self.__format_graph_name(t2)
        g.node(new)
        g.edge(self.__format_graph_name(t1), new)

    def get_prediction(self):
        return self.all_layers[-1]

    # def train_ae(self):
    #     loss = getLoss(x_train_node, y_predict)
