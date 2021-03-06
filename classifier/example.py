import  tensorflow as tf

'''
input > weight > hidden layer 1 (activation function) > weights > hidden l 2
(activation function) > weights > output layer
feed forward

compare output to intended output  > cost or loss function (cross entropy)
optimisation function (optimizer) > minimize cost (AdamOptimizer)

back and manipulate the weights - backpropagation

feed forward + backprop = epoch
'''

from tensorflow.example.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# 10 classes, 0 - 1
'''
one hot on element is on or hot
0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	#starting with something random
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl1))} # bias adds input_data * weights + bias
	
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}
	
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}
	
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal(n_classes))}

	# model (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) # treshold function

	l2 = tf.add(tf.matmul(data, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) # treshold function

	l3 = tf.add(tf.matmul(data, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3) # treshold function

	output = tf.matmul(output_layer, output_layer['weights']) + output_layer['biases']

	return output 