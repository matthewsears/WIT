# Neural Network (NN) in TensorFlow that learns the XOR gate
import tensorflow as tf

# XOR truth table
X_XOR = [[0,0],[0,1],[1,0],[1,1]]
Y_XOR = [[0],[1],[1],[0]]

# NN parameters
N_STEPS       = 250000
HIDDEN_NODES  = 2
INPUT_NODES   = 2
OUTPUT_NODES  = 1
LEARNING_RATE = 0.05

# Sigmoid activation function
def activate(nodes, weights):
    net = tf.matmul(nodes, weights) # net input is the dot product
    return tf.nn.sigmoid(net)

# NN I/O
x_nn = tf.placeholder(tf.float32, shape=[len(X_XOR),INPUT_NODES])
y_nn = tf.placeholder(tf.float32, shape=[len(X_XOR),OUTPUT_NODES])

# Randomized NN weights for input layer and hidden layer
w_i = tf.Variable(tf.random_uniform([INPUT_NODES, HIDDEN_NODES], 0, 1))
w_h = tf.Variable(tf.random_uniform([HIDDEN_NODES, OUTPUT_NODES], 0, 1))

# Forward Pass through the layers
hidden = activate(x_nn, w_i)
output = activate(hidden, w_h)

# Cost function (MSE) and Backpropagation
cost     = tf.reduce_mean(tf.square(Y_XOR - output))
backprop = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(N_STEPS):
    sess.run(backprop, feed_dict={x_nn: X_XOR, y_nn: Y_XOR})
    if i % 2500 == 0:
        print('Guess ', sess.run(output, feed_dict={x_nn: X_XOR, y_nn: Y_XOR}))
        print('Input Weights ', sess.run(w_i))
        print('Hidden Weights ', sess.run(w_h))
        print('Error ', sess.run(cost, feed_dict={x_nn: X_XOR, y_nn: Y_XOR}))
        print('\n')
