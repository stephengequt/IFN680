from keras.datasets import mnist
import numpy as np
import random
from keras import backend as K
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt


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


def create_pairs(x, digit_indices):  # pairs举一个正例和反例，labels为1 0 1 0 ...
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    print(n)  # 5420,891
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)  # 2-9
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def contrastive_loss(y_true, y_pred):  # 对比损失
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_base_network(input_shape):
    input = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    return keras.models.Model(input, x)


def create_CNN_model(shape):
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


def euclidean_distance(vects):  # 欧式距离
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print(shape1[0])
    return (shape1[0], 1)


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)  # 60000*28*28
# print(y_train)  # 60000
# print(x_test)  # 10000*28*28
# print(y_test)  # 10000
img_rows, img_cols = x_train.shape[1:3]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Concatenate the X and y data, then split the data into 80-20 proportio
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)
# X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2,
#                                                     random_state=42)
# Extract [0, 1, 8, 9] from training set and concatenate them to the test set
digits_group_1 = [2, 3, 4, 5, 6, 7]
digits_group_2 = [0, 1, 8, 9]
digits_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# all_digits = digits_to_keep + digits_to_be_removed
digits_group_1_index = np.array([True if i in digits_group_1 else False for i in y_all])

# keep the digits to be kept in training set only
x_group_1 = x_all[digits_group_1_index]
y_group_1 = y_all[digits_group_1_index]
x_group_2 = x_all[~digits_group_1_index]
y_group_2 = y_all[~digits_group_1_index]

# print(x_group_1.shape, y_group_1.shape)
# print(x_group_2.shape, y_group_2.shape)

x_train, x_test_group_1, y_train, y_test_group_1 = train_test_split(x_group_1, y_group_1, test_size=0.2,
                                                                    random_state=42)

x_test = np.append(x_test_group_1, x_group_2, axis=0)
y_test = np.append(y_test_group_1, y_group_2, axis=0)
print(x_test.shape)
print(y_test.shape)
# reorganize by groups
train_groups = [x_train[np.where(y_train == i)[0]] for i in np.unique(y_train)]
test_groups = [x_test[np.where(y_test == i)[0]] for i in np.unique(y_test)]
print('train groups:', [x.shape[0] for x in train_groups])
print('test groups:', [x.shape[0] for x in test_groups])

input_shape = x_train.shape[1:]
input_shape
# np.unique(y_to_keep)
# Append the removed data to the testing set

# print(revised_X_train)
# digit_indices = [np.where(revised_y_train == i)[0] for i in range(10)]  # y_train中值为i的下标值
digit_indices = [np.where(y_train == j)[0] for i, j in enumerate(digits_group_1)]
# TODO: digit_indices usage
digit_indices

train_pairs, train_y = create_pairs_train(x_train, digit_indices, digits_group_1)
train_pairs.shape
# print(tr_pairs)  # (108400,2,28,28)
# print(tr_y, len(tr_y))  # 108400,1 0 1 0交叉
# print(train_pairs.shape)
#
# mask = [True if i in digits_to_be_removed else False for i in revised_y_test]
# exp_2_X_test = revised_X_test[mask]
# exp_2_y_test = revised_y_test[mask]

# digit_indices = [np.where(revised_y_test == i)[0] for i in range(10)]
# digit_indices = [np.where(exp_2_y_test == j)[0] for i, j in enumerate(digits_to_keep)]
digit_indices = [np.where(y_test == j)[0] for i, j in enumerate(digits_all)]
digit_indices
# test_pairs, test_y = create_pairs(revised_X_test, digit_indices)

test_pairs, test_y = create_pairs_train(x_test, digit_indices, digits_all)
test_pairs.shape

# base_network = create_base_network(input_shape)
base_network = create_CNN_model(input_shape)
base_network.summary()

input_a = keras.Input(shape=input_shape)
input_b = keras.Input(shape=input_shape)

input_b

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = keras.layers.Lambda(euclidean_distance,  # 要实现的函数
                               output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = keras.models.Model([input_a, input_b], distance)
model.summary()

model.compile(loss=contrastive_loss, optimizer=keras.optimizers.RMSprop(), metrics=[accuracy])
# 拟合distance 和 1 0 1 0...

train_pairs[:, 0].shape
train_pairs[:, 1].shape
train_y.shape
test_pairs.shape
test_y.shape

history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
                    batch_size=128,
                    epochs=3,
                    validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y))

score = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_y, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

digit_indices = [np.where(y_group_2 == j)[0] for i, j in enumerate(digits_group_2)]
test_pairs, test_y = create_pairs_train(x_group_2, digit_indices, digits_group_2)
score = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_y, verbose=0)

digit_indices = [np.where(y_group_1 == j)[0] for i, j in enumerate(digits_group_1)]
test_pairs, test_y = create_pairs_train(x_group_1, digit_indices, digits_group_1)
score = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_y, verbose=0)

y_prediction = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
train_accuracy = compute_accuracy(test_y, y_prediction)

# y_prediction = model.predict([exp_2_pairs[:, 0], exp_2_pairs[:, 1]])
# test_accuracy = compute_accuracy(exp_2_y, y_prediction)

print('* Accuracy on training set: %0.2f%%' % (100 * train_accuracy))
# print('* Accuracy on test set: %0.2f%%' % (100 * test_accuracy))

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
