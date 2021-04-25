import numpy as np
import glob, math
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize


def load_dataSet(directory):
    imageDirectory = directory

    x_noise = list()
    x_clean = list()

    for name in glob.glob(imageDirectory):
        print(name)
        for image in glob.glob(name + "*.tif"):
            if (image.find('clean') != -1):
                x_clean.append(image)
                image = image.replace('clean', 'noise')
                x_noise.append(image)

    print(len(x_noise))
    print(len(x_clean))
    return x_noise, x_clean

class testSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]        
        return np.array([
            imread(file_name) / 255.
            for file_name in batch_x]), np.array([
            imread(file_name) / 255.
            for file_name in batch_y])
        