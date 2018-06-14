# Importing all the dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Defining the dataset
mnist = input_data.read_data_sets("MNIST_data/")

n_input = 28 * 28  # As the dimensions of all the images are 28*28

x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.int64, shape=[None])

n_hidden1 = 300  # 300 nodes in the first hidden layer
n_hidden2 = 100  # 100 nodes in the second hidden layer
n_output = 10  # 10 nodes in the output layer since we have to predict '10' digits


# 1st Hidden Layer
stddev1 = 2 / np.sqrt(n_input)
q1 = tf.truncated_normal([n_input, n_hidden1], stddev=stddev1)
w1 = tf.Variable(q1)
b1 = tf.Variable(tf.zeros([n_hidden1]))
hidden1 = tf.nn.elu(tf.matmul(x, w1) + b1) #Using Exponential Linear Unit(ELU) instead of ReLU as it outperforms all ReLU variants

# 2nd Hidden Layer
stddev2 = 2 / np.sqrt(n_hidden1)
q2 = tf.truncated_normal([n_hidden1, n_hidden2], stddev=stddev2)
w2 = tf.Variable(q2)
b2 = tf.Variable(tf.zeros([n_hidden2]))
hidden2 = tf.nn.elu(tf.matmul(hidden1, w2) + b2)

# Output Layer
stddev3 = 2 / np.sqrt(n_hidden2)
q3 = tf.truncated_normal([n_hidden2, n_output], stddev=stddev3)
w3 = tf.Variable(q3)
b3 = tf.Variable(tf.zeros([n_output]))
logits = tf.nn.softmax(tf.matmul(hidden2, w3) + b3)

# The loss function
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(entropy)

# Optimizer
learning_rate = 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 100
batch_size = 500

# Starting the Session
with tf.Session() as sess:
    init.run()  # Initializing all the global variables
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
        print(epoch, " Train accuracy: ", acc_train * 100)

# Test Accuracy
acc_test = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("Test accuracy: ", acc_test * 100)
