# Importing all the dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow .examples.tutorials.mnist import input_data  # Importing the MNIST dataset


# Defining the dataset
mnist = input_data.read_data_sets("MNIST_data/")

n_input = 28 * 28  # As the dimensions of all the images are 28*28
x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.int64, shape=[None])

n_output = 10  # As there are 10 digits (0-9)

#Weights and biases
stddev = 2 / np.sqrt(n_input)
q = tf.truncated_normal([n_input, n_output], stddev=stddev)
w = tf.Variable(q)
b = tf.Variable(tf.zeros([n_output]))

# The output layer
logits = tf.matmul(x, w) + b

# The loss function
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(entropy)

# Optimizer
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 400
batch_size = 50

# Starting the session
with tf.Session() as sess:
    init.run()  # Initializing all the global variables
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
        print(epoch, " Train accuracy: ", acc_train * 100)  # Printing the accuracy of the Train data batches individually

    # Accuracy of the Test data
    acc_test = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("Test accuracy: ", acc_test * 100)
