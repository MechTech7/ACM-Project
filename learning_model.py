import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import numpy as np

#https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4

#-----version info for later reference----------------
#TF version: 1.14.0
#Hub version: 0.7.0

#A note on image processing:
#The input images are expected to have color values in the range [0,1], 
#following the common image input conventions. 
#This module is suitable to be fine tuned.

#DatasetPackage is a class intended to package the train, validation, and test datasets
class DatasetPackage:
    def __init__(self, data_file):
        #The dataset package that takes in a file and initializes the train, validation and test sets
        self.IMAGE_WIDTH = 224 #the fixed image width that the feature extractor takes in
        #the data file can be a .npy file, really just a big matrix 


    def get_training_data(self):
        #return the training data and labels in a tuple
        train_X = self.pre_process(self.training_data)
        return (trainX, self.training_labels)

    def get_validation_data(self):
        #return validation data and labels in a tuple
        validate_X = self.pre_process(self.validate_data)
        return (validate_X, self.validate_labels)

    def get_test_data(self):
        #return the test data and labels as a tuple
        test_X = self.pre_process(self.test_data)
        return (test_X, self.test_labels)

#We'll use a ResNet trained on ImageNet as a feature extractor
#It takes in 224x224 Images so we'll have to do a bit of downscaling
feature_extraction_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4", trainable=False)
#feature extraction outputs a feature vector of length 2048

#Model Definition:
model = keras.Sequential()odel progress can be saved during—and after—training. This means a model can resume where it left off and avoid long training times. Saving also means you can share your model and others can recreate your work. When publishing research models and techniques, most machine learning practitioners share:

code to create the model, and train it
model.add(feature_extraction_layer) #add the feature extraction layer

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation=None)) #"regression" layer


train_X, train_Y =

