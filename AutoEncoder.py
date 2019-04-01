# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:44:44 2019

@author: sacha
"""

    




#%% AutoEncodeur TensorFlow (les variables sont en Majuscules)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

s = tf.InteractiveSession()
 #%%  création des données (exemple MNIST) 
 
import gzip
f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
largeur=28
longueur=28
num_images = 50000
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size)

ff=gzip.open('train-labels-idx1-ubyte.gz','r')
ff.read(8)
buff = ff.read(num_images)
labels = np.frombuffer(buff, dtype=np.uint8).astype(np.float32)
labels = labels.reshape(num_images,1)

temp=np.zeros((num_images,10))
for k in range(num_images):
    temp[k][int(labels[k])]=1
labels=temp

train=data.reshape(num_images, image_size* image_size)/256

X_train=train[0:40000]
X_test=train[40000:50000]


#%% l'autoencodeur. 
#il faut avoir déja défini X_train et X_test
#largeur de l'image*longueur de l'image = Num_classes


## Defining various initialization parameters for 784-512-256-10 MLP model
Num_classes = X_train.shape[1]
Num_features = X_train.shape[1]
Num_output = X_train.shape[1]
Num_layers_0 = 256
Num_layers_1 = 64
Num_layers_2 = 16
Num_layers_3 = 64
Num_layers_4 = 256
starter_learning_rate = 0.001
regularizer_rate = 0.1

# Placeholders for the input data
Input_X = tf.placeholder('float32',shape =(None,Num_features),name="Input_X")
Input_y = tf.placeholder('float32',shape = (None,Num_classes),name='Input_Y')
## for dropout layer
keep_prob = tf.placeholder(tf.float32)


#initialising biases and weights
## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
Weights_0 = tf.Variable(tf.random_normal([Num_features,Num_layers_0]))
Bias_0 = tf.Variable(tf.random_normal([Num_layers_0]))
Weights_1 = tf.Variable(tf.random_normal([Num_layers_0,Num_layers_1], stddev=(1/tf.sqrt(float(Num_layers_0)))))
Bias_1 = tf.Variable(tf.random_normal([Num_layers_1]))
Weights_2 = tf.Variable(tf.random_normal([Num_layers_1,Num_layers_2], stddev=(1/tf.sqrt(float(Num_layers_1)))))
Bias_2 = tf.Variable(tf.random_normal([Num_layers_2]))
Weights_3 = tf.Variable(tf.random_normal([Num_layers_2,Num_layers_3], stddev=(1/tf.sqrt(float(Num_layers_2)))))
Bias_3 = tf.Variable(tf.random_normal([Num_layers_3]))
Weights_4 = tf.Variable(tf.random_normal([Num_layers_3,Num_layers_4], stddev=(1/tf.sqrt(float(Num_layers_3)))))
Bias_4 = tf.Variable(tf.random_normal([Num_layers_4]))
Weights_5 = tf.Variable(tf.random_normal([Num_layers_4,Num_output], stddev=(1/tf.sqrt(float(Num_layers_4)))))
Bias_5 = tf.Variable(tf.random_normal([Num_output]))

## Initializing network
Hidden_output_0 = tf.nn.tanh(tf.matmul(Input_X,Weights_0)+Bias_0)
Hidden_output_0_0 = tf.nn.dropout(Hidden_output_0, keep_prob)
Hidden_output_1 = tf.nn.tanh(tf.matmul(Hidden_output_0_0,Weights_1)+Bias_1)
Hidden_output_1_1 = tf.nn.dropout(Hidden_output_1, keep_prob)
Hidden_output_2 = tf.nn.tanh(tf.matmul(Hidden_output_1_1,Weights_2)+Bias_2)
Hidden_output_2_2 = tf.nn.dropout(Hidden_output_2, keep_prob)
Hidden_output_3 = tf.nn.tanh(tf.matmul(Hidden_output_2_2,Weights_3)+Bias_3)
Hidden_output_3_3 = tf.nn.dropout(Hidden_output_3, keep_prob)
Hidden_output_4 = tf.nn.tanh(tf.matmul(Hidden_output_3_3,Weights_4)+Bias_4)
Hidden_output_4_4 = tf.nn.dropout(Hidden_output_4, keep_prob)
Predicted_y = tf.nn.tanh(tf.matmul(Hidden_output_4_4,Weights_5) + Bias_5)
Clearer_y=tf.nn.relu(Predicted_y)

## Defining the loss function (quadratic)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))
Loss=tf.losses.mean_squared_error(Input_X,Clearer_y)

## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)

## Adam optimzer for finding the right weight
Optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss,var_list=[Weights_0,Weights_1,Weights_2,Weights_3,Weights_4,Weights_5,Bias_0,Bias_1,Bias_2,Bias_3,Bias_4,Bias_5])

## Metrics definition
#correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
Accuracy = 1-Loss

## Training parameters
batch_size = 32
epochs=7
dropout_prob = 0.9
training_accuracy = []
training_loss = []
testing_accuracy = []
s.run(tf.global_variables_initializer())
for epoch in range(epochs):    
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0,X_train.shape[0],batch_size):
        s.run(Optimizer, {Input_X: X_train[arr[index:index+batch_size]], Input_y: X_train[arr[index:index+batch_size]], keep_prob:dropout_prob})
    training_accuracy.append(s.run(Accuracy, feed_dict= {Input_X:X_train, Input_y: X_train,keep_prob:1}))
    ## Evaluation of model
    testing_accuracy.append(s.run(Accuracy, feed_dict= {Input_X:X_test, Input_y: X_test,keep_prob:1}))
    print("Epoch:{" + str( epoch) + "}, Train accuracy :" + str(round(training_accuracy[epoch]*10000)/100) + "% Test accuracy :" + str(round(testing_accuracy[epoch]*10000)/100)+"%")
#%% visualisation

numero=np.random.randint(0,len(X_train))
exin=X_train[numero:numero+1]
exout=s.run(Clearer_y, {Input_X: exin, keep_prob:1})
exin=np.reshape(exin,(largeur,longueur))
exout=np.reshape(exout,(largeur,longueur))
plt.subplot(1,2,1)
plt.imshow(data[numero])
plt.subplot(1,2,2)
plt.imshow(exout)