import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class DatasetPackage:
    def __init__(self, data_file):
        #The dataset package that takes in a file and initializes the train, validation and test sets
        self.IMAGE_WIDTH = 224 #the fixed image width that the feature extractor takes in
        in_data = np.load(data_file)
        self.x_data = in_data[:, :224, :, :] #The pixel values of each tile: (check to make sure this is the right shape)
        self.y_data = in_data[:, 224, 0, 0] #the GDP of each tile

        #Train/Validation/Test split
        self.x_data = self.x_data.astype('float32')
        self.y_data = self.y_data.astype('float32')

        num_examples = in_data.shape[0]
        #80/10/10 split
        train_count = int(0.8 * num_examples)
        valid_count = int((num_examples - train_count) / 2)
        
        valid_index = train_count + valid_count
        self.training_data = self.x_data[:train_count, :, :, :]
        self.training_labels = self.y_data[:train_count]

        self.valid_data = self.x_data[train_count:valid_index, :, :, :]
        self.valid_labels = self.y_data[train_count:valid_index]

        self.test_data = self.x_data[valid_index:-1, :, :, :]
        self.test_labels = self.y_data[valid_index:-1]


        #the data file can be a .npy file, really just a big matrix 

    def get_overfit_test(self):
        return (self.training_data[:2, :, :, :], self.training_labels[:2])
    def get_training_data(self):
        #return the training data and labels in a tuple
        train_X = self.training_data
        return (train_X, self.training_labels)

    def get_validation_data(self):
        #return validation data and labels in a tuple
        validate_X = self.validate_data
        return (validate_X, self.validate_labels)

    def get_test_data(self):
        #return the test data and labels as a tuple
        test_X = self.test_data
        return (test_X, self.test_labels)

print(type(keras.initializers.RandomNormal()))

conv_layer_1 = layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation=tf.keras.activations.relu, 
                                kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.RandomNormal())

conv_layer_2 = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding="same", activation=tf.keras.activations.relu, 
                                kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.RandomNormal())

conv_layer_3 = layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation=tf.keras.activations.relu, 
                                kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.RandomNormal())

conv_layer_4 = layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="same", activation=tf.keras.activations.relu, 
                                kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.RandomNormal())

inputs = keras.Input(shape=(224, 224, 3, ), name="input_images")

conv_1 = conv_layer_1(inputs)
conv_1 = layers.BatchNormalization()(conv_1)

conv_2 = conv_layer_2(conv_1)
conv_2 = layers.BatchNormalization()(conv_2)

conv_3 = conv_layer_3(conv_2)
conv_3 = layers.BatchNormalization()(conv_3)

conv_4 = conv_layer_4(conv_3)
conv_4 = layers.BatchNormalization()(conv_4)

conv_4 = layers.Flatten()(conv_4)
x = layers.Dense(128, activation=tf.keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.RandomNormal())(conv_4)
x = layers.Dense(64, activation=tf.keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.RandomNormal())(x)
x = layers.Dense(1, activation=tf.keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(), bias_initializer=keras.initializers.RandomNormal())(x)
model = keras.Model(inputs=inputs, outputs=x)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(1e-5), 
                loss=keras.losses.mean_squared_error,
                metrics=['mae'])

data_file = "compiledData.npy"#dataset filename
data_pack = DatasetPackage(data_file)
train_x, train_y = data_pack.get_training_data()


model.fit(x=train_x, y=train_y, epochs=1200, batch_size=32)
model.save("saved_models/model_epoch_0.hd5")
