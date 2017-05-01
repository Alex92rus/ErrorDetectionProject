import tensorflow as tf
import numpy as np
import random
import re
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
import fce_api as fd
import collections
import class_util
import matplotlib.pyplot as plt

# Generate tain and test data
# X, Y = make_classification(n_samples=50000, n_features=10, n_informative=8,
#                     n_redundant=0, n_clusters_per_class=2)
# Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class
# X, X_test, Y, Y_test = train_test_split(X, Y)

#--------- My part data init ---------
# generating the word windows, including the spaces.
def feed_windows_with_spaces(_data, _window_size):
    windows = []
    for sentence, errors in _data:
        tokens = re.split(r'(\s+)', sentence)
        word_window_size = min(len(tokens), _window_size)
        for i in range(0, len(tokens) - word_window_size + 1):
            window_tuple = (tokens[i:i + word_window_size], )
            window_range = range(round(i / 2), round((i + word_window_size) / 2))
            for error in errors:
                if error[0] in window_range or error[1] in window_range:
                    if len(window_tuple) < 2:
                        window_tuple = window_tuple + (has_error, )
            if len(window_tuple) == 1:
                window_tuple = window_tuple + (no_error, )
            windows.append(window_tuple)
    return windows


# generating the word windows only with the tokens, excluding the spaces.
def feed_windows_only_tokens(_data, _window_size):
    windows = []
    max_length = 0
    for sentence, errors in _data:
        tokens = sentence.split()
        if len(tokens) > max_length:
            max_length = len(tokens)
        word_window_size = min(len(tokens), _window_size)
        for i in range(0, len(tokens) - word_window_size + 1):
            window_tuple = (tokens[i:i + word_window_size], )
            window_range = range(i, i + word_window_size)
            for error in errors:
                if error[0] in window_range or error[1] in window_range:
                    if len(window_tuple) < 2:
                        window_tuple = window_tuple + (has_error, )
            if len(window_tuple) == 1:
                window_tuple = window_tuple + (no_error, )
            windows.append(window_tuple)
    return windows

# converting the windows to their numerical representation
def convert_to_np(_feed_windows, vocab, train=True):
    feed_windows_np = np.full([len(_feed_windows), window_size], PAD)
    labels = np.full([len(_feed_windows), 2], PAD, dtype=np.int64)
    for i, window in enumerate(_feed_windows):
        window_length = len(window[0])
        seq = np.full(window_length, 0)
        for j, token in enumerate(window[0]):
            if token not in vocab:
                if train:
                    vocab[token] = len(vocab)#np.random.rand(100)[1]
                else:
                    vocab[token] = vocab['<OOV>']
            seq[j] = vocab[token]
        feed_windows_np[i, 0:window_length] = seq
        if window[1] == has_error:
            labels[i] = [0, 1]
        else:
            labels[i] = [1, 0]
    return feed_windows_np, labels, vocab

# general variables
window_size = 50

# classes
has_error = 1
no_error = 0

# training data
training_data = fd.extract_data_from_tsv('fce-public.train.original.tsv')

# dev data
dev_data = fd.extract_data_from_tsv('fce-public.dev.original.tsv')

# amt golden data
amt_golden = fd.extract_data('amt_data_sets/fce_amt_experiment_two.max.rasp.m2')

# amt non-expert data
amt_non_expert = fd.extract_data('amt_data_sets/fce_amt_experiment_best_result.m2')

# omitting the spaces
feed_windows = feed_windows_only_tokens(training_data, window_size)


dev_feed_windows = feed_windows_only_tokens(dev_data, window_size)


labels = []
PAD = 0.0
OOV = 1.0
display_step = 1


vocab = {'<PAD>': PAD, '<OOV>': OOV}

# test:calculating the errors and no errors
num_right = len([x for x in feed_windows if x[1] == 0])
num_errors = len([x for x in feed_windows if x[1] == 1])

# # oversampling: get a constant gap between the classes (250 here)
# for i in range(len(feed_windows)):
#     if feed_windows[i][1] == 1:
#         feed_windows.append(feed_windows[i])
#         num_errors = num_errors + 1
#     if num_errors == num_right - 19:
#         break

# feed_windows = feed_windows[:42000]
feed_windows_np, train_labels, vocab = convert_to_np(feed_windows, vocab)

dev_feed_windows_np, dev_labels, vocab = convert_to_np(dev_feed_windows, vocab, train=False)

X, Y = make_classification(n_samples=len(feed_windows_np), n_features=3, n_informative=2,
                    n_redundant=0, n_clusters_per_class=2)
Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class

print(X[1])
print(feed_windows_np[1])
X = feed_windows_np
Y = train_labels
X_dev = dev_feed_windows_np
Y_dev = dev_labels
#--------- My part data init ---------

# Parameters
k = 0.5
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.0001 #tf.train.inverse_time_decay(0.01, global_step, k, 40)
decay = 40
training_epochs = 500
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 256 # 3rd layer number of features
n_input = 50 # Number of feature
n_classes = 2 # Number of classes to predict

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# arrays for plotting
train_accuracies = []
dev_accuracies = []
train_errors = []
dev_errors = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    accuracy_val = 0
    epoch = 0
    while accuracy_val < 0.96:
        avg_cost = 0.
        c = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
        np.random.shuffle(c)
        X2 = c[:, :X.size//len(X)].reshape(X.shape)
        Y2 = c[:,X.size//len(X):].reshape(Y.shape)
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X2, total_batch)
        Y_batches = np.array_split(Y2, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # batch_y.shape = (batch_y.shape[0], 1)
            # Run optimization op (backprop) and cost op (to get loss value)

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            train_errors.append(avg_cost)
        # Display logs per epoch step
        epoch = epoch + 1
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % epoch, "cost=", \
                "{:.9f}".format(avg_cost))
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_val = accuracy.eval({x: X, y: Y})
        train_accuracies.append(accuracy_val)
        print("Train Accuracy:", accuracy_val)

        dev_accuracy_val = accuracy.eval({x: X_dev, y: Y_dev})
        dev_accuracies.append(dev_accuracy_val)
        print("Dev Accuracy:", dev_accuracy_val)

        dev_error = sess.run(cost, feed_dict={x: X_dev, y: Y_dev})
        dev_errors.append(dev_error)
        print("dev cost: ", dev_error)

        #----------
        predicted = sess.run(tf.argmax(pred, 1), {x: X})
        counter = collections.Counter(predicted)
        print(counter)

        t = epoch % (decay + 1)
    print ("Optimization Finished!")

    plt.figure(1)
    plt.plot(train_errors)
    plt.xlabel('epochs')
    plt.ylabel('train error')

    plt.figure(2)
    plt.plot(train_accuracies)
    plt.xlabel('epochs')
    plt.ylabel('train accuracy')

    plt.figure(3)
    plt.plot(dev_accuracies)
    plt.xlabel('epochs')
    plt.ylabel('dev accuracy')

    plt.figure(4)
    plt.plot(dev_errors)
    plt.xlabel('epochs')
    plt.ylabel('dev errors')

    plt.show()
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: X, y: Y}))

    predictions = sess.run(tf.argmax(pred, 1), feed_dict={x: X})
    actual_labels = np.argmax(Y, 1)
    cm_train = class_util.create_confusion_matrix(actual_labels, predictions)
    evaluation_data_frame = class_util.full_evaluation_table(cm_train)
    print(evaluation_data_frame)

    predictions = sess.run(tf.argmax(pred, 1), feed_dict={x: X_dev})
    actual_labels = np.argmax(Y_dev, 1)
    cm_train = class_util.create_confusion_matrix(actual_labels, predictions)
    evaluation_data_frame = class_util.full_evaluation_table(cm_train)
    print(evaluation_data_frame)
    global result
    result = tf.argmax(pred, 1).eval({x: X, y: Y})