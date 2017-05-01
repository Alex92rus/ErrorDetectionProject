import tensorflow as tf
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import fce_api as fd
import collections
import sklearn.metrics as metrics
import pandas as pd

# This code is done with the help of examples from
# https://github.com/uclmr/stat-nlp-book 
# https://github.com/aymericdamien/TensorFlow-Examples.77
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


def remove_sentences(train, to_remove):
    """
       Removes to_remove sentences from train
       Args:
           train: training data
           to_remove: sentences to be removed
    """
    result = [sentence for sentence in train if sentence[0] not in [x[0] for x in to_remove]]
    return result

#
# with_amt = [0.49984825, 0.50531191, 0.51843983, 0.51449388, 0.50398391, 0.51733953, 0.51760513, 0.51233119, 0.51126879, 0.51900893]
# without_amt = [0.49453634, 0.51282442, 0.50307328, 0.51623917, 0.51707393, 0.51479739, 0.50701928, 0.51305205, 0.51081347, 0.52098197]
with_amt = [0.47837483617300131, 0.50533430006732616, 0.51502972913455181, 0.51432799522782546, 0.48609782539378865, 0.52649683122081392, 0.48168637924456309, 0.52649734810918025, 0.5045406685969992, 0.49343310934636531]
without_amt = [0.50834847796456162, 0.46571213262999245, 0.48407985343357135, 0.42614977807158377, 0.49662100733377779, 0.47269368644377791, 0.51318893685372946, 0.50741918388977214, 0.44499476348402867, 0.38091539196940727]

def plot_classifier_metrics(train_accuracies, amt_accuracies, label='accuracy'):
    train_accuracies_range =list(range(10, 101, 10))
    amt_accuracies_range =list(range(10, 101, 10))
    plt.plot(train_accuracies_range, train_accuracies, 'bs', label='only train')
    plt.plot(amt_accuracies_range, amt_accuracies, 'g^', label='train plus AMT annotations')
    plt.xlabel('percent training data')
    plt.ylabel(label)
    plt.ylim([0.0, 1.0])
    plt.xlim([8, 102])
    plt.legend(loc='upper right', numpoints=1)
    plt.xticks(train_accuracies_range)
    plt.grid()
    plt.show()

plot_classifier_metrics(without_amt, with_amt, label=r'$F_{0.5}$')

# general variables
window_size = 5

# classes
has_error = 1
no_error = 0

# # training data
# training_data = fd.extract_data_from_tsv('fce-public.train.original.tsv')
#
# # dev data
# dev_data = fd.extract_data_from_tsv('fce-public.dev.original.tsv')

# amt golden data
amt_golden = fd.extract_data('amt_data_sets/fce_amt.experiment_two.max.rasp.m2')

# amt non-expert data
amt_non_expert = fd.extract_data('amt_data_sets/fce_amt.experiment_two.max.rasp.m2')

# omitting the spaces
e_feed_windows = feed_windows_only_tokens(amt_golden, window_size)


ne_feed_windows = feed_windows_only_tokens(amt_non_expert, window_size)


labels = []
PAD = 0.0
OOV = 1.0
display_step = 1


vocab = {'<PAD>': PAD, '<OOV>': OOV}

# test:calculating the errors and no errors
num_right = len([x for x in e_feed_windows if x[1] == 0])
num_errors = len([x for x in e_feed_windows if x[1] == 1])

# # oversampling: get a constant gap between the classes (250 here)
# for i in range(len(feed_windows)):
#     if feed_windows[i][1] == 1:
#         feed_windows.append(feed_windows[i])
#         num_errors = num_errors + 1
#     if num_errors == num_right - 19:
#         break

# feed_windows = feed_windows[:42000]
e_feed_windows_np, e_train_labels, vocab = convert_to_np(e_feed_windows, vocab)

ne_feed_windows_np, ne_labels, vocab = convert_to_np(ne_feed_windows, vocab, train=False)


print(e_feed_windows_np[1])
X = e_feed_windows_np
Y = e_train_labels
X_ne = ne_feed_windows_np
Y_ne = ne_labels
#--------- My part data init ---------

# Parameters
k = 0.5
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.0001 # tf.train.inverse_time_decay(0.01, global_step, k, 40)
decay = 40
training_epochs = 500
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 256 # 3rd layer number of features
n_input = 5 # Number of feature
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


def session_cycle(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, total_batches=9, batch_size=10):
    # arrays for plotting
    train_accuracies = []
    dev_accuracies = []
    train_errors = []
    dev_errors = []
    test_accuracy_val = 0
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        accuracy_val = 0
        epoch = 0
        dev_accuracy_val = 0
        early_stop_counter = 0

        while accuracy_val < 0.86:
            avg_cost = 0.
            c = np.c_[X_train.reshape(len(X_train), -1), Y_train.reshape(len(Y_train), -1)]
            np.random.shuffle(c)
            X2 = c[:, :X_train.size // len(X_train)].reshape(X_train.shape)
            Y2 = c[:, X_train.size // len(X_train):].reshape(Y_train.shape)
            total_batch = int(len(X_train) / batch_size)
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
            accuracy_val = accuracy.eval({x: X_train, y: Y_train})
            train_accuracies.append(accuracy_val)
            print("Train Accuracy:", accuracy_val)

            dev_accuracy_val = accuracy.eval({x: X_dev, y: Y_dev})
            dev_accuracies.append(dev_accuracy_val)
            print("Dev Accuracy:", dev_accuracy_val)

            dev_error = sess.run(cost, feed_dict={x: X_dev, y: Y_dev})
            if len(dev_accuracies) > 1 and len(dev_errors) > 1 and dev_error > dev_errors[-1]:
                early_stop_counter += 1
                if early_stop_counter > 4 or epoch > 45:
                    test_accuracy_val = accuracy.eval({x: X_test, y: Y_test})
                    print("Test Accuracy:", test_accuracy_val)

                    break
            else:
                early_stop_counter = 0
            dev_errors.append(dev_error)
            print("dev cost: ", dev_error)

            t = epoch % (decay + 1)
        print("Optimization Finished!")
        # ----------
        predicted = sess.run(tf.argmax(pred, 1), {x: X_test})
        real_labels = np.argmax(Y_test, axis = 1)
        counter = collections.Counter(predicted)
        precision = metrics.precision_score(real_labels, predicted)
        recall = metrics.recall_score(real_labels, predicted)
        f05 = (1 + 0.5**2) * precision * recall/(0.5**2*precision + recall) if precision * recall > 0 else 0.0
        print(counter)

    return test_accuracy_val, f05

# Initializing the variables
init = tf.initialize_all_variables()

cross_validation_accuracies = []

vocab_session = {'<PAD>': PAD, '<OOV>': OOV}
amt_train_batch = fd.extract_data_from_tsv('amt_data_sets/fce_public.best.amt.tsv')
training_data = fd.extract_data_from_tsv('fce-public.train.original.tsv')
dev_data = fd.extract_data_from_tsv('fce-public.dev.original.tsv')
test_data = fd.extract_data_from_tsv('fce-public.test.original.tsv')
print('len training data:', len(training_data))
training_data = remove_sentences(training_data, amt_train_batch)
print('len training data:', len(training_data))
one_tenth = len(training_data) // 10
train_amt_accuracies = []
train_accuracies = []
train_amt_f05s= []
train_f05s = []
for i in range(1, 11):
    vocab_session = {'<PAD>': PAD, '<OOV>': OOV}
    indexes = random.sample(range(0, len(training_data)), one_tenth * i)
    train_batch = [sentence[1] for sentence in enumerate(training_data) if sentence[0] in indexes]
    train_only_feed_windows_session = feed_windows_only_tokens(train_batch, window_size)
    ne_feed_windows_session = feed_windows_only_tokens(train_batch + amt_train_batch, window_size)
    dev_feed_widows_session = feed_windows_only_tokens(dev_data, window_size)
    test_feed_widows_session = feed_windows_only_tokens(test_data, window_size)
    e_feed_windows_np_session, e_labels_session, vocab_session = convert_to_np(train_only_feed_windows_session, vocab_session)
    dev_feed_windows_np_session, dev_labels_session, vocab_session = convert_to_np(dev_feed_widows_session, vocab_session, train=False)
    test_feed_windows_np_session, test_labels_session, vocab_session = convert_to_np(test_feed_widows_session, vocab_session, train=False)
    train_accuracy, train_f05 = session_cycle(e_feed_windows_np_session, e_labels_session, dev_feed_windows_np_session, dev_labels_session, test_feed_windows_np_session, test_labels_session)
    train_accuracies.append(train_accuracy)
    train_f05s.append(train_f05)
    vocab_session = {'<PAD>': PAD, '<OOV>': OOV}
    ne_feed_windows_np_session, ne_labels_session, vocab_session = convert_to_np(ne_feed_windows_session, vocab_session)
    dev_feed_windows_np_session, dev_labels_session, vocab_session = convert_to_np(dev_feed_widows_session, vocab_session, train=False)
    train_amt_accuracy, train_amt_f05 = session_cycle(ne_feed_windows_np_session, ne_labels_session, dev_feed_windows_np_session, dev_labels_session, test_feed_windows_np_session, test_labels_session)
    train_amt_accuracies.append(train_amt_accuracy)
    train_amt_f05s.append(train_amt_f05)

print('Accuracy With AMT data', train_amt_accuracies)
print('Accuracy Without AMT data', train_accuracies)
print('F0.5 With AMT data', train_amt_f05s)
print('F0.5 Without AMT data', train_f05s)
plot_classifier_metrics(train_accuracies, train_amt_accuracies)
plot_classifier_metrics(train_f05s, train_amt_f05s, label=r'$F_{0.5}$')
df_data = {}
df_data['train accuracies'] = train_accuracies
df_data['train and AMT accuracies'] = train_amt_accuracies
df_data['train F0.5'] = train_f05s
df_data['train and AMT F0.5'] = train_amt_f05s
df = pd.DataFrame(data=df_data)
df.to_csv(path_or_buf='statistics/classifier_comparison_45.csv', columns=['train accuracies', 'train and AMT accuracies',
                                                                       'train F0.5', 'train and AMT F0.5'])
