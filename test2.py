from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as pd
import tensorflow as tf

DATA_DIR = 'C:\\Users\\User\\Desktop\\Mango\\MangoRecongnition\\MyTrainingData'

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    dtype='float32'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    directory=DATA_DIR + '/train',
    target_size=(224, 224),
    batch_size=16,
    # subset='training'
)

validation_generator = test_datagen.flow_from_directory(
    directory=DATA_DIR + '/dev',
    target_size=(224, 224),
    batch_size=16,
    # subset='validation'
)

data = train_generator.next()

print(data)