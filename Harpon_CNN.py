#from conf import *

import tensorflow as tf
import numpy as np

sess = tf.Session()

img_size=160
image_size=3
num_channels=1

filter_size_conv1=5
num_filters_conv1=32
filter_size_conv2=5
num_filters_conv2=16
fc_layer_size=128

classes = ['up', 'down', 'still']
num_classes = len(classes)
 
 
# validation split
validation_size = 0.2
 
# batch size
batch_size = 1

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
    
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    # We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
 
    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
 
    layer += biases
 
    # We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
 
    return layer
    
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer
    
    
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
 
    return layer
    
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
 
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
 
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_flat = create_flatten_layer(layer_conv2)
 
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)
 
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)
                     
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)                     
                     
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)



def front_prop(state):
    
    image = state
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = np.array(image).reshape(1, image_size,image_size,num_channels)
    graph = tf.get_default_graph()
    
    y_pred = graph.get_tensor_by_name("y_pred:0")
    
    # Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 2)) 
    
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    return result

def back_prop(data): #data is a list of lists : [States, actions, Q-values]
 
    for i in range(len(data)):
        
        
        x_batch = data[0][i]
        
        real_y = front_prop(x_batch)
        
        y_true_batch = real_y.copy()
        y_true_batch[data[1][i]]=data[2][i]
 
        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
 
        session.run(optimizer, feed_dict=feed_dict_tr)
 
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, 'CNN-Pong') 
 
 
    
