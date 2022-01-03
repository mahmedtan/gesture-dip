import os
import sys
from cv2 import cv2
import numpy as np
import pickle
import filteration as filter
import tensorflow as tf


class Preprocess():

    def __init__(self):
        self.training_data = []
        self.training_labels = []
        self.train_sizes = []
        self.testing_data = []
        self.testing_labels = []
        self.test_sizes =[]


    def initial_process(self,folder_name):

        sub_folders = os.listdir(folder_name)
        letter_mapping = {}
        count = 0

        for sub in sub_folders:
            letter_mapping[sub] = count
            count += 1

        for sub in sub_folders:
            all_images = os.listdir("./"+folder_name+"/"+sub)
            total_num = len(all_images)
            training_num = int(total_num*0.85)
            testing_num = total_num - training_num
            for i in range(total_num):
                img_path = "./"+folder_name+"/"+sub+"/"+all_images[i]
                # print(sub, i)
                image = cv2.imread(img_path, )
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image = filter.laplace(image)
                image = filter.sobely(image)
                # filter.plot(image,image2)
                flattened = image.flatten()/255.0
                if i < testing_num:
                    if (len(flattened)==4096):
                        self.testing_data.append(flattened)
                        self.testing_labels.append(letter_mapping[sub])
                        self.test_sizes.append(len(flattened))
                else:
                    if (len(flattened)==4096):
                        self.training_data.append(flattened)
                        self.training_labels.append(letter_mapping[sub])
                        self.train_sizes.append(len(flattened))

    def get_data(self):
        self.testing_data = tf.transpose(tf.reshape(self.testing_data,(-1,1,64,64)),perm=[0,2,3,1])
        self.testing_labels = tf.one_hot(self.testing_labels,32,dtype=tf.int32)
        self.training_data = tf.transpose(tf.reshape(self.training_data,(-1,1,64,64)),perm=[0,2,3,1])
        self.training_labels = tf.one_hot(self.training_labels,32,dtype=tf.int32)
        print(self.testing_data.shape, self.testing_labels.shape, self.training_data.shape, self.training_labels.shape)
        return self.testing_data, self.testing_labels, self.training_data, self.training_labels
