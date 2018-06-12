# Importing all the dependencies
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

# Defining the dataset
mnist = input_data.read_data_sets("MNIST_data/")


x = tf.placeholder(tf.float32, shape=(None, 28 * 28))  # As the dimensions of all the images are 28*28
y = tf.placeholder(tf.int64, shape=(None))


# Creating the hidden layers
n_hidden1 = 300  # 300 nodes in the first hidden layer
n_hidden2 = 100  # 100 nodes in the second hidden layer
n_output = 10  # 10 nodes in the output layer since we have to predict '10' digits

hidden1 = fully_connected(x, n_hidden1)
hidden2 = fully_connected(hidden1, n_hidden2)
logits = fully_connected(hidden2, 10, activation_fn=None)

# The loss function
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(entropy)

# Optimizer
learning_rate = 0.03
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
        acc_test = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(epoch, " Train accuracy: ", acc_train * 100, " Test accuracy: ", acc_test * 100)
