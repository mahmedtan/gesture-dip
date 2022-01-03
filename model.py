from __future__ import absolute_import
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_array_ops import pad
import tensorflow as tf
from preprocess import Preprocess

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 32
        self.loss_list = []
        # hyperparameters
        self.learning_rate = 0.001
        self.alpha = 0.2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.first_layer = 64
        self.second_layer = 128
        self.third_layer = 256
        self.dense_size = 512

        #block1
        self.conv_layer_1 = tf.keras.layers.Conv2D(self.first_layer,3,strides=(2,2),padding="SAME", activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.max_pool_1 = tf.keras.layers.MaxPool2D(3,strides=(2,2),padding="SAME")

        #block2
        self.conv_layer_2 = tf.keras.layers.Conv2D(self.second_layer,3,strides=(2,2),padding="SAME", activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.max_pool_2 = tf.keras.layers.MaxPool2D(2,strides=(1,1),padding="SAME")

        #block3
        self.conv_layer_3 = tf.keras.layers.Conv2D(self.third_layer,3,strides=(2,2),padding="SAME",activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.Dense_1 = tf.keras.layers.Dense(self.dense_size,activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.Dense_2 = tf.keras.layers.Dense(self.num_classes)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        out_1 = self.batch_norm_1(self.conv_layer_1(inputs))
        pooled_1 = self.max_pool_1(out_1)

        out_2 = self.batch_norm_2(self.conv_layer_2(pooled_1))
        pooled_2 = self.max_pool_2(out_2)

        out_3 = self.batch_norm_3(self.conv_layer_3(pooled_2))

        flattened = self.flatten(out_3)

        dense_1 = self.dropout_1(self.Dense_1(flattened))
        return self.Dense_2(dense_1)

    def loss(self, logits, labels):

        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        NOTE: DO NOT EDIT
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    indices = tf.random.shuffle([x for x in range(len(train_inputs))])
    train_inputs = tf.gather(train_inputs,indices)
    train_labels = tf.gather(train_labels,indices)
    for i in range(0,len(train_inputs),model.batch_size):
        batched_inputs = train_inputs[i:i+model.batch_size]
        batched_inputs = tf.image.random_flip_left_right(batched_inputs)
        batched_labels = train_labels[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            predictions = model.call(batched_inputs)
            loss = model.loss(predictions,batched_labels)
            model.loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    acc = 0
    for i in range(0,len(test_inputs),model.batch_size):
        batched_inputs = test_inputs[i:i+model.batch_size]
        batched_labels = test_labels[i:i+model.batch_size]
        predictions = model.call(batched_inputs)
        acc += model.accuracy(predictions,batched_labels)
    return acc/(len(test_inputs)/model.batch_size)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def visualize_accuracy(accs):
    """
    Uses Matplotlib to visualize accuracies of our model.
    :param accs: list of accuracy data stored from test.
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(accs))]
    plt.plot(x, accs)
    plt.title('Accuracy per batch')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = predicted_labels[ind]
            al = np.argmax(image_labels[ind], axis=0)
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.
    CS1470 students should receive a final accuracy
    on the testing examples for cat and dog of >=70%.
    CS2470 students should receive a final accuracy
    on the testing examples for cat and dog of >=75%.
    :return: None
    '''
    print("Pre processing started")
    pre = Preprocess()
    pre.initial_process("ArASL_Database_54K_Final")
    test_inputs, test_labels, train_inputs, train_labels = pre.get_data()
    print(test_inputs.shape, test_labels.shape, train_inputs.shape, train_labels.shape)
    print("Pre processing done!")
    num_epochs = 25
    acc_list = []
    model = Model()
    accuracy = test(model,test_inputs,test_labels)
    print("Accuracy for epoch", 0 , accuracy)
    for i in range(num_epochs):
        print("EPOCH -", i+1)
        train(model,train_inputs,train_labels)
        accuracy = test(model,test_inputs,test_labels)
        print("Accuracy for epoch", i+1 , accuracy)
        acc_list.append(accuracy)
    accuracy = test(model,test_inputs,test_labels)
    print("Accuracy", accuracy)
    # visualize_results(train_inputs[0:10],model.call(train_inputs[0:10,:,:,:]),train_labels[0:10])
    visualize_loss(model.loss_list)
    visualize_accuracy(acc_list)
    return


if __name__ == '__main__':
    main()