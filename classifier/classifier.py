import tensorflow as tf
import sklearn as sk
import sklearn.metrics as skm
import numpy as np
import math
import fce_api as fd
import re
import matplotlib.pyplot as plt

# variables
# data
data = fd.extract_data('fce_train.gold.max.rasp.old_cat.m2')

window_size = 3

labels = []
PAD = 0
display_step = 1

# classes
has_error = 1
no_error = 0

# learning variables
learning_rate = 0.01
epochs = 50


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
    for sentence, errors in _data:
        tokens = sentence.split()
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

feed_windows = feed_windows_only_tokens(data, window_size)
# feed_windows = feed_windows[:200]
# changing the training data
print('feed_windows: ', '%d' % len(feed_windows), 'sentences: ', '%d' % len(data))

# feed_windows = feed_windows[:42000]

print('ready')

vocab = {'<PAD>': PAD}
feed_windows_np = np.full([len(feed_windows), window_size], PAD)
labels = np.full([len(feed_windows), 1], 2)

batch_size = len(feed_windows_np)//2

for i, window in enumerate(feed_windows):
    window_length = len(window[0])
    seq = np.full(window_length, 0)
    for j, token in enumerate(window[0]):
        if token not in vocab:
            vocab[token] = len(vocab)
        seq[j] = vocab[token]
    feed_windows_np[i, 0:window_length] = seq
    if window[1] == has_error:
        labels[i] = 1
    else:
        labels[i] = 0

x = tf.placeholder(tf.float32, [None, window_size]) # data with the size of the window currently set to 6
y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([window_size, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model
model = tf.nn.sigmoid(tf.matmul(x, W))  # Sigmoid

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model), reduction_indices=1))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



print(len(feed_windows_np))

# Splitting to train and test
train = feed_windows_np[:len(feed_windows_np)//2 - ((len(feed_windows_np)//2) % batch_size)]
train_labels = labels[:len(feed_windows_np)//2 - ((len(feed_windows_np)//2) % batch_size)]
test = feed_windows_np[(len(feed_windows_np) * 2) // 3:]

total_batches = int(math.floor(len(train) / batch_size))


test_labels = labels[(len(feed_windows_np) * 2) // 3:]
print('len train: ', '%d' % len(train), 'len test: ', '%d' % len(test))

ones_train = len([x for x in train_labels if x[0] == 1])
ones_train_proportion = ones_train / len(train_labels)
print('Ones prop in training set ', '%09f' % ones_train_proportion)

ones_test = len([x for x in test_labels if x[0] == 1])
ones_test_proportion = ones_train / len(test_labels)
print('Ones prop in test set ', '%09f' % ones_test_proportion)

avg_costs = []
with tf.Session() as sess:

    # Initializing the variables
    init = tf.initialize_all_variables()
    sess.run(init)

    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.
        for i in range(total_batches):
            feed_dict = {
                x: train[i * batch_size:(i + 1) * batch_size],
                y: labels[i * batch_size:(i + 1) * batch_size]
            }
            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
            # sess.run(optimizer, feed_dict=feed_dict)
            # updating the loss
            avg_cost += c / total_batches
        if (epoch + 1) % display_step == 0:
            print ('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
            avg_costs.append(avg_cost)
    print ('Optimisation Finished!')

    plt.plot(avg_costs)
    plt.show()

    # Training accuracy
    correct_prediction = tf.equal(model, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_val = accuracy.eval({x: train, y: train_labels})
    print('Accuracy on training data: ', accuracy_val)


    pred_y = sess.run(model, feed_dict={x: train})
    eps = pow(10, -6)
    pred_y_trunc = np.asarray([[1.] if 1. - eps < x <= 1. else [0.] for x in pred_y], dtype=np.float32)
    print('Calculated in another way: ', np.mean(pred_y_trunc == train_labels))

    print("Precision: ", skm.precision_score(train_labels, pred_y_trunc))
    print("Recall: ", skm.recall_score(train_labels, pred_y_trunc))
    print("f1_score: ", skm.f1_score(train_labels, pred_y_trunc))

    # Test accuracy
    print('Accuracy on test data:', accuracy.eval({x: test, y: test_labels}))