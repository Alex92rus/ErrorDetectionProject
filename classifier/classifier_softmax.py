import tensorflow as tf
import numpy as np
import math
import fce_api as fd
import re

# variables
# data
data = fd.extract_data_from_tsv('fce-public.train.original.tsv')

window_size = 3
PAD = 0
display_step = 1

# classes
no_error = 'no_error'

# learning variables
learning_rate = 0.01
epochs = 10
batch_size = 128


# generating the word windows, including the spaces.
def feed_windows_with_spaces(_data, _window_size, _error_types):
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
                        window_tuple = window_tuple + (error[2], )
                        if error[2] not in _error_types:
                            _error_types[error[2]] = len(_error_types)
            if len(window_tuple) == 1:
                window_tuple = window_tuple + (no_error, )
            windows.append(window_tuple)
    return windows


# generating the word windows only with the tokens, excluding the spaces.
def feed_windows_only_tokens(_data, _window_size, _error_types):
    windows = []
    for sentence, errors in _data:
        tokens = sentence.split()
        word_window_size = min(len(tokens), _window_size)
        for i in range(0, len(tokens) - word_window_size + 1):
            window_tuple = (tokens[i:i + word_window_size], )
            window_range = range(i, i + word_window_size)
            for error in errors:
                if error[0] in window_range or error[1] in window_range:
                    if len(window_tuple) < 2:
                        window_tuple = window_tuple + (error[2], )
                        if error[2] not in _error_types:
                            _error_types[error[2]] = len(_error_types)
            if len(window_tuple) == 1:
                window_tuple = window_tuple + (no_error, )
            windows.append(window_tuple)
    return windows

error_types = {no_error: 0}

feed_windows = feed_windows_only_tokens(data, window_size, error_types)
error_types_length = len(error_types)
# feed_windows = feed_windows[:200]
# changing the training data
print('feed_windows: ', '%d' % len(feed_windows), 'sentences: ', '%d' % len(data))

# feed_windows = feed_windows[:42000]

print('ready')

vocab = {'<PAD>': PAD}
feed_windows_np = np.full([len(feed_windows), window_size], PAD)
labels = np.full([len(feed_windows), error_types_length], 0)

# creating the labels and vectorizing the word windows.
for i, window in enumerate(feed_windows):
    window_length = len(window[0]) 
    seq = np.full(window_length, 0)
    for j, token in enumerate(window[0]):
        if token not in vocab:
            vocab[token] = len(vocab)
        seq[j] = vocab[token]  
    feed_windows_np[i, 0:window_length] = seq
    # One hot enumeration
    labels[i][error_types[window[1]]] = 1

x = tf.placeholder(tf.float32, [None, window_size]) # data with the size of the window currently set to 6
y = tf.placeholder(tf.float32, [None, error_types_length])

W = tf.Variable(tf.zeros([window_size, error_types_length]))
b = tf.Variable(tf.zeros([error_types_length]))

# Construct model
model = tf.nn.softmax(tf.matmul(x, W) + b) # Sigmoid

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))

#cost = tf.reduce_mean(y*tf.log(tf.clip_by_value(model,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


predict = tf.argmax(model, 1)

print(len(feed_windows_np) + b)

# Splitting to train and test
train = feed_windows_np[:len(feed_windows_np)//2 - ((len(feed_windows_np)//2) % batch_size)]
train_labels = labels[:len(feed_windows_np)//2 - ((len(feed_windows_np)//2) % batch_size)]
test = feed_windows_np[(len(feed_windows_np) * 2) // 3:]
test_labels = labels[(len(feed_windows_np) * 2) // 3:]
total_batches = int(math.floor(len(train) / batch_size))

avg_costs = []

with tf.Session() as sess:
    # Initializing the variables
    init = tf.initialize_all_variables()
    sess.run(init)
    y_true = np.argmax(test_labels, 1)

    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0
        for start, end in zip(range(0, len(train), batch_size), range(batch_size, len(train) + 1, batch_size)):
            _, c = sess.run([optimizer, cost], feed_dict={x: train[start:end], y: train_labels[start:end]})
            avg_cost += c/total_batches
        print('avg_cost: ' + str(avg_cost))
        y_pred = sess.run(predict, feed_dict={x: test})
        print(epoch, np.mean(y_true == y_pred))

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: test, y: test_labels}))