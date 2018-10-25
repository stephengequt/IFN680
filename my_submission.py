import numpy as np

import tensorflow as tf
from tensorflow import keras

# from tensorflow.contrib import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # outfile = "mnist_dataset.npz"
    # np.savez(outfile, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # TODO: x and y need to be separated by digits requirement
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    #
    outfile = "mnist_dataset.npz"
    np.savez(outfile, x_all=x_all, y_all=y_all)


def pre_processing():
    '''
    Save the arrays x_train, y_train, x_test and y_test
    into a single npz file named 'mnist_dataset.npz'
    '''

    # TODO:
    with np.load('mnist_dataset.npz') as npzfile:
        x_all = npzfile['x_all']
        y_all = npzfile['y_all']

        print('x_all : ', x_all.shape, x_all.dtype)
        print('y_all : ', y_all.shape, y_all.dtype)

        fig = plt.figure()
        for i in range(20, 30):
            img = x_all[i]
            fig.add_subplot(2, 5, i - 19)
            plt.title(y_all[i])
            plt.imshow(img, cmap='gray')
        plt.show()

    # x_all =



    # use default data
    with np.load('mnist_dataset.npz') as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']  # INSERT YOUR CODE HERE
        x_test = npzfile['x_test']  # INSERT YOUR CODE HERE
        y_test = npzfile['y_test']  # INSERT YOUR CODE HERE

    # x_train = x_train.values.reshape(-1, 28, 28, 1).astype('float')/256
    x_train = x_train.astype('float') / 255
    # x_test = x_test.values.reshape(-1, 28, 28, 1).astype('float')/256
    x_test = x_test.astype('float') / 255
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    print('Training', x_train.shape, x_train.max())
    print('Testing', x_test.shape, x_test.max())
    digi_array = np.array([2, 3, 4])
    # reprganize by groups
    train_groups = [x_train[np.where(y_train == i)[0]] for i in digi_array]
    train_group_array = np.array(train_groups)
    test_groups = [x_test[np.where(y_test == i)[0]] for i in np.unique(y_test)]
    print('train groups:', [x.shape[0] for x in train_groups])
    print('train groups:', train_groups.shape)
    print('test groups:', [x.shape[0] for x in test_groups])
    return train_groups, test_groups, x_test, y_test, x_train, y_train


def gen_random_batch(in_groups, batch_halfsize=8):
    out_img_a, out_img_b, out_score = [], [], []
    all_groups = list(range(len(in_groups)))
    for match_group in [True, False]:
        group_idx = np.random.choice(all_groups, size=batch_halfsize)
        out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]
        if match_group:
            b_group_idx = group_idx
            out_score += [1] * batch_halfsize
        else:
            # anything but the same group
            non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
            b_group_idx = non_group_idx
            out_score += [0] * batch_halfsize

        out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]

    return np.stack(out_img_a, 0), np.stack(out_img_b, 0), np.stack(out_score, 0)


load_data()
train_groups, test_groups, x_test, y_test, x_train, y_train = pre_processing()

pv_a, pv_b, pv_sim = gen_random_batch(train_groups, 3)
fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize=(12, 6))
for c_a, c_b, c_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, m_axs.T):
    ax1.imshow(c_a[:, :])
    ax1.set_title('Image A')
    ax1.axis('off')
    ax2.imshow(c_b[:, :])
    ax2.set_title('Image B\n Similarity: %3.0f%%' % (100 * c_d))
    ax2.axis('off')

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout

img_in = Input(shape=x_train.shape[1:], name='FeatureNet_ImageInput')
n_layer = img_in

# TODO:
for i in range(2):
    n_layer = Conv2D(8 * 2 ** i, kernel_size=(3, 3), activation='linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = Conv2D(16 * 2 ** i, kernel_size=(3, 3), activation='linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = MaxPool2D((2, 2))(n_layer)
n_layer = Flatten()(n_layer)
n_layer = Dense(32, activation='linear')(n_layer)
n_layer = Dropout(0.5)(n_layer)
n_layer = BatchNormalization()(n_layer)
n_layer = Activation('relu')(n_layer)
feature_model = Model(inputs=[img_in], outputs=[n_layer], name='FeatureGenerationModel')
feature_model.summary()
