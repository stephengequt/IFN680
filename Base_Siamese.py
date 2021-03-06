from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.keras.api.keras as keras

import random
from keras.models import Sequential
from keras.layers.core import *
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split


def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss vg
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    x, y = inputs
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_pairs_train(x, digit_indices, train_digits):
    # Create empty list of pairs and labels to be appended
    pairs = []
    labels = []

    # calculate the min number of training sample of each digit in training set
    min_sample = [len(digit_indices[d]) for d in range(len(train_digits))]

    # calculate the number of pairs to be created
    n = min(min_sample) - 1

    # Looping over each digits in the train_digits
    for d in range(len(train_digits)):

        # Create n pairs of same digits and then create the same amount of pairs for the different digits
        for i in range(n):
            # Create a pair of same digits:
            # retrieve the index of a pair of same digit
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # Append the image pair of same digits to the pair list
            pairs += [[x[z1], x[z2]]]

            # Create a pair of different digits
            # First create a randome integer rand falls between (1, len(train_digits))
            # let dn be (d+rand) % len(train_digit) so that dn will distinct from d
            # and that is guaranteed to be a different digits
            rand = random.randrange(1, len(train_digits))
            dn = (d + rand) % len(train_digits)

            # Use the dn and d to create a pair of different digits
            # the append it to the pair list
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]

            # Append the corresponding label value for the true and false pairs of image created
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


#
# def create_negative_pairs(x, digit_indices, set_1, set_2):
#     # Create empty list of pairs and labels to be appended
#     pairs = []
#     labels = []
#     all_digits = set_1 + set_2
#
#     # calculate the min number of training sample of each digit in training set
#     min_sample = [len(digit_indices[d]) for d in range(len(all_digits))]
#
#     # calculate the number of pairs to be created
#     n = min(min_sample)
#
#     # Looping over each digits in the train_digits
#     for d in range(len(set_1)):
#
#         # Create n pairs of same digits and then create the same amount of pairs for the different digits
#         for i in range(n):
#             # Create a pair of different digits
#             # First create a randome integer rand falls between (0, len(set_2))
#             # let dn be len(set_1) + random.randrange(0, len(set_2)) so that dn
#             # will be the index of digits in set_2
#             # and that is guaranteed to be a different digits
#             dn = len(set_1) + random.randrange(0, len(set_2))
#
#             # Use the dn and d to create a pair of different digits
#             # the append it to the pair list
#             z1, z2 = digit_indices[d][i], digit_indices[dn][i]
#             pairs += [[x[z1], x[z2]]]
#
#             # Append the corresponding label value for the true and false pairs of image created
#             labels += [0]
#
#     return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    return keras.models.Model(input, x)


def create_custom_model(shape):
    custom_model = keras.models.Sequential()
    # Layer 0 should be Conv2D with 32 filters and 3x3 kernel size. Use 'relu' for the activation option
    custom_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape))
    # Layer 1 should be another Conv2D with 64 filters and 3x3 kernel size. Use again 'relu' for the activation option.
    custom_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # Layer 2 should be a MaxPooling2D with pool_size 2x2.
    custom_model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    # Layer 3 should be a Dropout with probability 0.25.
    custom_model.add(keras.layers.Dropout(0.25))
    # Layer 4 should be a Flatten layer.
    custom_model.add(keras.layers.Flatten())
    # Layer 5 should be a Dense layer with 128 neurons and 'relu' as the activation function.
    custom_model.add(keras.layers.Dense(128, activation='relu'))
    # Layer 6 should be a Dropout with probability 0.5.
    custom_model.add(keras.layers.Dropout(0.5))
    # Layer 7 should be a Dense layer with 10 neurons and 'softmax' as the activation function.
    custom_model.add(keras.layers.Dense(10, activation='softmax'))
    return custom_model


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


#
# def display_image(image):
#     plt.imshow(image, cmap=plt.get_cmap('gray'))
#     plt.show()


epochs = 2

# Call the Keras.datasets.mnist API and split the data into training and test set
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

img_rows, img_cols = X_train.shape[1:3]
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# Ensure the data type is float instead of int
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the matrix
X_train /= 255
X_test /= 255

# Each input data should be in the form of 28 by 28 matrix
input_shape = X_train.shape[1:]
img_rows, img_cols = X_train.shape[1:3]
custom_input_shape = (img_rows, img_cols, 1)

# Concatenate the X and y data, then split the data into 80-20 proportio
X_all = np.append(X_train, X_test, axis=0)
y_all = np.append(y_train, y_test, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2,
                                                    random_state=42)

# Extract [0, 1, 8, 9] from training set and concatenate them to the test set
digits_to_keep = [2, 3, 4, 5, 6, 7]
digits_to_be_removed = [0, 1, 8, 9]
all_digits = digits_to_keep + digits_to_be_removed
mask = np.array([True if i in digits_to_keep else False for i in y_train])

# keep the digits to be kept in training set only
revised_X_train = X_train[mask]
revised_y_train = y_train[mask]

# Append the removed data to the testing set
revised_X_test = np.append(X_test, X_train[~mask], axis=0)
revised_y_test = np.append(y_test, y_train[~mask], axis=0)

###############################################################################
#                        Training Pairs Creation
###############################################################################

# Now the training set contains only the digits in digits_to_keep[2,3,4,5,6,7]
# To train a siamese NN, we need to create samples of true pairs and false pairs
# where the True pairs are pairs of same digits (2,2), (3,3), (4,4), ..., (7, 7)
# whereas the False pairs are pairs of different digits (2, 3), (4, 6), (6, 7)..

# First, create a digit-index matrix where each row is the indices of a particular digits in the training set
#
# digit 2 [ ind, ind, ind,...............ind]
# digit 3 [ ind, ind, ind,...............ind]
# digit 4 [ ind, ind, ind,...............ind]
# digit 5 [ ind, ind, ind,...............ind]
# digit 6 [ ind, ind, ind,...............ind]
# digit 7 [ ind, ind, ind,...............ind]
digit_indices = [np.where(revised_y_train == j)[0] for i, j in enumerate(digits_to_keep)]

# Create training pairs (tr_pairs) and the corresponding y value (tr_y)
# Note the training set contains only the digits_to_keep [2,3,4,5,6,7]
tr_pairs, tr_y = create_pairs_train(revised_X_train, digit_indices, digits_to_keep)
"""
for i in range(5):
    display_image(tr_pairs[2000*i+1][0])
    plt.show()
    display_image(tr_pairs[2000*i+1][1])
    plt.show()    

    print(tr_y[i*2000+1])
 """
###############################################################################
#                Create testing set for experiment 1
# -----------------------------------------------------------------------------
#              Experiment 1 [2,3,4,5,6,7] vs [2,3,4,5,6,7]
###############################################################################
# Construct a subset from test set that only contains the digits of [2,3,4,5,6,7]
# then create pairs of testing data and the corresponding labels
mask = [True if i in digits_to_keep else False for i in revised_y_test]
exp_1_X_test = revised_X_test[mask]
exp_1_y_test = revised_y_test[mask]

# create the digit-index matrix where each row is the indices of a particular digits in the training set
digit_indices = [np.where(exp_1_y_test == j)[0] for i, j in enumerate(digits_to_keep)]
"""
for j in range(len(digits_to_keep)): 

    for i in range(10):
        display_image(exp_1_X_test[digit_indices[j][-i]])
        plt.show()

"""

# Create testing pairs and the corresponding y value for experiment 1
exp_1_pairs, exp_1_y = create_pairs_train(exp_1_X_test, digit_indices, digits_to_keep)
"""
for i in range(5):
    display_image(exp_1_pairs[3005*i][0])
    plt.show()
    display_image(exp_1_pairs[3005*i][1])
    plt.show()    

    print(exp_1_y[i*3005])
"""
###############################################################################
#                Create testing set for experiment 2
# -----------------------------------------------------------------------------
#              Experiment 2 [0,1,8,9] vs [0,1,8,9]
###############################################################################

# Construct a subset from test set that only contains the digits of [0,1,8,9]
# then create pairs of testing data and the corresponding labels
mask = [True if i in digits_to_be_removed else False for i in revised_y_test]
exp_2_X_test = revised_X_test[mask]
exp_2_y_test = revised_y_test[mask]

# create the digit-index matrix where each row is the indices of a particular digits in the training set
digit_indices = [np.where(exp_2_y_test == j)[0] for i, j in enumerate(digits_to_be_removed)]

# Create testing pairs and the corresponding y value for experiment 2
exp_2_pairs, exp_2_y = create_pairs_train(exp_2_X_test, digit_indices, digits_to_be_removed)

###############################################################################
#                Create testing set for experiment 3
# -----------------------------------------------------------------------------
#              Experiment 3 [2,3,4,5,6,7] vs [0,1,8,9]
###############################################################################

# create the digit-index matrix where each row is the indices of a particular digits in the training set
digit_indices = [np.where(revised_y_test == j)[0] for i, j in enumerate(all_digits)]

# Create testing pairs and the corresponding y value for experiment 3
exp_3_pairs, exp_3_y = create_pairs_train(revised_X_test, digit_indices, all_digits)
"""
for i in range(10):
    display_image(exp_3_pairs[1001*i][0])
    plt.show()
    display_image(exp_3_pairs[1001*i][1])
    plt.show()    

    print(exp_3_y[i*801])
"""

###############################################################################
#                     Defining Neural Network
# -----------------------------------------------------------------------------
#
#
###############################################################################

# Defining Siamese neural network
# base_network = create_base_network(input_shape)
custom_network = create_custom_model(custom_input_shape)

# input_a = keras.layers.Input(shape=input_shape)
# input_b = keras.layers.Input(shape=input_shape)
custom_input_a = keras.layers.Input(shape=custom_input_shape)
custom_input_b = keras.layers.Input(shape=custom_input_shape)

# reuse the instance of the neural network so that the weights of the neuralnet
# will be shared accross the two branches
# processed_a = base_network(input_a)
# processed_b = base_network(input_b)

custom_processed_a = custom_network(custom_input_a)
custom_processed_b = custom_network(custom_input_b)

# distance = keras.layers.Lambda(euclidean_distance,
#                                output_shape=eucl_dist_output_shape)([processed_a, processed_b])

custom_distance = keras.layers.Lambda(euclidean_distance,
                                      output_shape=eucl_dist_output_shape)([custom_processed_a, custom_processed_b])

# model = keras.models.Model([input_a, input_b], distance)
custom_model = keras.models.Model([custom_input_a, custom_input_b], custom_distance)
# train
rms = keras.optimizers.RMSprop()
# model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
custom_model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])


###############################################################################
#                   Model fitting and evaluation
# ------------------------------------------------------------------------------
#
###############################################################################
def fit_model(model, tr_pairs, tr_y, test_pairs, test_y):
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs,
              validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y))

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
    te_acc = compute_accuracy(test_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    return tr_acc, te_acc


def custom_fit_model(model, tr_pairs, tr_y, test_pairs, test_y):
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=128,
                        epochs=10,
                        validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y))

    # history.history

    # # Plot training & validation accuracy values
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
    te_acc = compute_accuracy(test_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    return tr_acc, te_acc, history


#
# exp_1_train_accuracy, exp_1_test_accuracy = fit_model(custom_model, tr_pairs, tr_y, exp_1_pairs, exp_1_y)
# exp_2_train_accuracy, exp_2_test_accuracy = fit_model(custom_model, tr_pairs, tr_y, exp_2_pairs, exp_2_y)
# exp_3_train_accuracy, exp_3_test_accuracy = fit_model(custom_model, tr_pairs, tr_y, exp_3_pairs, exp_3_y)

# exp_1_train_accuracy, exp_1_test_accuracy = custom_fit_model(custom_model, tr_pairs, tr_y, exp_1_pairs, exp_1_y)
exp_2_train_accuracy, exp_2_test_accuracy, history = custom_fit_model(custom_model, tr_pairs, tr_y, exp_2_pairs,
                                                                      exp_2_y)
# exp_3_train_accuracy, exp_3_test_accuracy = custom_fit_model(custom_model, tr_pairs, tr_y, exp_3_pairs, exp_3_y)
print('**********************************************************************')
# print('* Accuracy on training set: %0.2f%%' % (100 * exp_1_train_accuracy))
# print('* Accuracy on test set: %0.2f%%' % (100 * exp_1_test_accuracy))

print('* Accuracy on training set: %0.2f%%' % (100 * exp_2_train_accuracy))
print('* Accuracy on test set: %0.2f%%' % (100 * exp_2_test_accuracy))

# print('* Accuracy on training set: %0.2f%%' % (100 * exp_3_train_accuracy))
# print('* Accuracy on test set: %0.2f%%' % (100 * exp_3_test_accuracy))
# #
# tr_pairs[:, 0].shape
# tr_y.shape
# exp_1_pairs.shape
#
# custom_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           batch_size=128,
#           epochs=epochs,
#           validation_data=([exp_1_pairs[:, 0], exp_1_pairs[:, 1]], exp_1_y))
#
# # compute final accuracy on training and test sets
# y_pred = custom_model.predict([tr_pairs[:, 0], tr_pairs[:, 1], 1])
# tr_acc = compute_accuracy(tr_y, y_pred)
# y_pred = custom_model.predict([exp_1_pairs[:, 0], exp_1_pairs[:, 1],1])
# te_acc = compute_accuracy(exp_1_y, y_pred)
#
# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

#
# history = custom_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#                            batch_size=128,
#                            epochs=epochs,
#                            validation_data=([exp_1_pairs[:, 0], exp_1_pairs[:, 1]], exp_1_y))

# history.history

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#
# from keras.utils import plot_model
#
# plot_model(custom_model, to_file='model.png')
