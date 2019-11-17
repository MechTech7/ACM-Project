import tensorflow as tf
import numpy as np
from tensorflow import keras

saved_model = keras.models.load_model("./saved_models/model_epoch_0_hd5")

def predict(tileDir):
    npyTile = np.load(tileDir)
    return (saved_model.predict(npyTile) * 1000)