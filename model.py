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
     
     

        out_1 = self.batch_norm_1(self.conv_layer_1(inputs))
        pooled_1 = self.max_pool_1(out_1)

        out_2 = self.batch_norm_2(self.conv_layer_2(pooled_1))
        pooled_2 = self.max_pool_2(out_2)

        out_3 = self.batch_norm_3(self.conv_layer_3(pooled_2))

        flattened = self.flatten(out_3)

        dense_1 = self.dropout_1(self.Dense_1(flattened))
        return self.Dense_2(dense_1)

    def loss(self, logits, labels):

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))

    def accuracy(self, logits, labels):
     
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):

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

    acc = 0
    for i in range(0,len(test_inputs),model.batch_size):
        batched_inputs = test_inputs[i:i+model.batch_size]
        batched_labels = test_labels[i:i+model.batch_size]
        predictions = model.call(batched_inputs)
        acc += model.accuracy(predictions,batched_labels)
    return acc/(len(test_inputs)/model.batch_size)


def visualize_loss(losses):
  
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def visualize_accuracy(accs):

    x = [i for i in range(len(accs))]
    plt.plot(x, accs)
    plt.title('Accuracy per batch')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):

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
    visualize_loss(model.loss_list)
    visualize_accuracy(acc_list)
    return


if __name__ == '__main__':
    main()