# Importing all the dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Defining the dataset
mnist = input_data.read_data_sets("MNIST_data/")


# In TensorFlow, each input image is typically represented as a 3D tensor of shape [height, width, channels]. A minibatch
# is represented as a 4D tensor of shape [mini-batch size, height, width, channels]. The weights of a
# convolutional layer are represented as a 4D tensor of shape [fh, fw, fn, fn']. The bias terms of a convolutional layer are simply
# represented as a 1D tensor of shape [fn'].


x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])  # shape in CNNs is always None x height x width x color channels
# Here 'x' is a minibatch
y = tf.placeholder(tf.int64, shape=[None])


# Step 1. Convolutional + Max Pooling Layers

# Convolution
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))  # shape is filter height x filter width x channels x output layers
b1 = tf.constant(0.1, shape=[32])  # shape is output layers
conv1 = tf.nn.relu(tf.nn.conv2d(input=x, filter=w1, strides=[1, 1, 1, 1], padding="SAME") + b1) # Convolution with the input as 'x' and 'w1' as the filter, padding="SAME" implies that the dimensions of the output layer are the same as those of the input layer
# Max Pooling
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Step 2. Convolutional + Max Pooling Layers

# Convolution
w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b2 = tf.constant(0.1, shape=[64])
conv2 = tf.nn.relu(tf.nn.conv2d(input=pool1, filter=w2, strides=[1, 1, 1, 1], padding="SAME") + b2)
# Max Pooling
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Step 3. 1st Fully Connected Layer
w3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b3 = tf.constant(0.1, shape=[1024])
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, w3) + b3)


# Step 4. Regularization using Dropout
keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)


# Step 5. Output Layer
w4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b4 = tf.constant(0.1, shape=[10])
logits = tf.matmul(fc1_drop, w4) + b4

# The loss function
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(entropy)

# Optimizer
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)
            trainingInputs = batch[0].reshape([batch_size, 28, 28, 1])
            trainingLabels = batch[1]
            sess.run(training_op, feed_dict={x: trainingInputs, y: trainingLabels, keep_prob: 0.5})
        acc_train = accuracy.eval(feed_dict={x: trainingInputs, y: trainingLabels, keep_prob: 1.0})
        print(epoch, " Train accuracy: ", acc_train * 100)
        
    # Printing the Test data accuracy
    acc_test = accuracy.eval(feed_dict={x: mnist.test.images.resahpe([-1, 28, 28, 1]), y: mnist.test.labels, keep_prob: 1.0})
    print("Test accuracy: ", acc_test * 100)
