import tensorflow as tf
import sklearn as sk
import random
import sklearn.metrics as skm
import numpy as np
import math
import fce_api as fd
import re
import matplotlib.pyplot as plt

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

def convert_to_np(_feed_windows, vocab):
    feed_windows_np = np.full([len(_feed_windows), window_size], PAD)
    labels = np.full([len(_feed_windows), 2], 2)
    for i, window in enumerate(_feed_windows):
        window_length = len(window[0])
        seq = np.full(window_length, 0)
        for j, token in enumerate(window[0]):
            if token not in vocab:
                vocab[token] = len(vocab)
            seq[j] = vocab[token]
        feed_windows_np[i, 0:window_length] = seq
        if window[1] == has_error:
            labels[i] = [1, 0]
        else:
            labels[i] = [0, 1]
    return feed_windows_np, labels, vocab

# variables
window_size = 3

labels = []
PAD = 0
display_step = 1

# classes
has_error = 1
no_error = 0

# learning variables
learning_rate = 0.01
epochs = 20


# training data
training_data = fd.extract_data_from_tsv('fce-public.train.original.tsv')

# dev data
dev_data = fd.extract_data_from_tsv('fce-public.dev.original.tsv')

feed_windows = feed_windows_only_tokens(training_data, window_size)

def testing(offset_bias):

    num_right = len([x for x in feed_windows if x[1] == 0])
    num_errors = len([x for x in feed_windows if x[1] == 1])
    # oversampling: get the errors once more
    for i in range(len(feed_windows)):
        if feed_windows[i][1] == 1:
            feed_windows.append(feed_windows[i])
            num_errors = num_errors +1
        if (num_errors == num_right + offset_bias):
            break

    # shuffling
    random.shuffle(feed_windows)

    print('feed_windows: ', '%d' % len(feed_windows), 'sentences: ', '%d' % len(training_data))

    vocab = {'<PAD>': PAD}
    # feed_windows = feed_windows[:42000]
    feed_windows_np, train_labels, vocab = convert_to_np(feed_windows, vocab)



    dev_feed_windows = feed_windows_only_tokens(dev_data, window_size)

    # shuffling
    random.shuffle(dev_feed_windows)

    # convert to np arrays for classifier
    dev_feed_windows, dev_labels, vocab = convert_to_np(dev_feed_windows, vocab)


    batch_size = len(feed_windows_np)//2

    x = tf.placeholder(tf.float32, [None, window_size]) # data with the size of the window currently set to 6
    y = tf.placeholder(tf.float32, [None, 2])

    W = tf.Variable(tf.zeros([window_size, 1]))
    b = tf.Variable(tf.zeros([2]))

    # Construct model
    model = tf.nn.softmax(tf.matmul(x, W) + b)  # Sigmoid

    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(model), reduction_indices=1))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



    print(len(feed_windows_np))

    # Splitting to train and test
    # train = feed_windows_np[:len(feed_windows_np)//2 - ((len(feed_windows_np)//2) % batch_size)]
    # train_labels = labels[:len(feed_windows_np)//2 - ((len(feed_windows_np)//2) % batch_size)]
    # test = feed_windows_np[(len(feed_windows_np) * 2) // 3:]
    # test_labels = labels[(len(feed_windows_np) * 2) // 3:]


    total_batches = int(math.floor(len(feed_windows_np) / batch_size))

    print('len train: ', '%d' % len(feed_windows_np), 'len dev: ', '%d' % len(dev_feed_windows))

    ones_train = len([x for x in train_labels if x[0] == 1])
    ones_train_proportion = ones_train / len(train_labels)
    print('Errors in training set ', '%09f' % ones_train_proportion)

    ones_test = len([x for x in dev_labels if x[0] == 1])
    ones_test_proportion = ones_train / len(dev_labels)
    print('Errors prop in test set ', '%09f' % ones_test_proportion)

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
                    x: feed_windows_np[i * batch_size:(i + 1) * batch_size],
                    y: train_labels[i * batch_size:(i + 1) * batch_size]
                }
                _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
                # sess.run(optimizer, feed_dict=feed_dict)
                # updating the loss
                avg_cost += c / total_batches
            if (epoch + 1) % display_step == 0:
                print ('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
                avg_costs.append(avg_cost)
        print ('Optimisation Finished!')

        #plt.plot(avg_costs)
        #plt.show()

        # Training accuracy
        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_val = accuracy.eval({x: feed_windows_np, y: train_labels})
        print('Accuracy on training data: ', accuracy_val)


        predicted = tf.argmax(model,1)
        y_pred_one_hot = sess.run(predicted, feed_dict={x:feed_windows_np})
        pred_y = sess.run(model, feed_dict={x: feed_windows_np})

        eps = pow(10, -6)
        # pred_y_trunc = np.asarray([[1.] if 1. - eps < x <= 1. else [0.] for x in pred_y], dtype=np.float32)
        # print('Calculated in another way: ', np.mean(pred_y_trunc == train_labels))
        #
        # print("Precision: ", skm.precision_score(train_labels, pred_y_trunc))
        # print("Recall: ", skm.recall_score(train_labels, pred_y_trunc))
        # print("f1_score: ", skm.f1_score(train_labels, pred_y_trunc))

        # Test accuracy
        print('Accuracy on dev data:', accuracy.eval({x: dev_feed_windows, y: dev_labels}))
        return y_pred_one_hot

labelshistory = []
for i in range(-1000, 1000, 1) :
    y_labels = testing(i)
    print('error:' + str(len([x for x in y_labels if x == 0])) + '\n no error:' + str(len([x for x in y_labels if x == 1])))
    labelshistory.append(len([x for x in y_labels if x == 0]))
plt.plot(labelshistory)
plt.show()


# equal_classes = np.full([100, window_size], PAD)
# selected_labels = np.full([100], 2)
# error_count = 47
# errors = 0
# clean = 0
#
# # some shuffling
# for i, window in enumerate(feed_windows_np):
#     if errors == error_count and  clean == (100 - error_count):
#         break
#     if (labels[i] == 1 and errors < error_count):
#         equal_classes[errors + clean] = feed_windows_np[i]
#         selected_labels[errors + clean] = labels[i]
#         errors = errors + 1
#     if (labels[i] == 0 and clean < 100 - error_count):
#         equal_classes[errors + clean] = feed_windows_np[i]
#         selected_labels[errors + clean] = labels[i]
#         clean = clean + 1
#
# labels = selected_labels
# feed_windows_np = equal_classes
