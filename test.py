import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers


trainPath = 'Data/train/'
devPath   = 'Data/dev/'

trainCSV = 'Data/train.csv'
devCSV   = 'Data/dev.csv'

trainDF = pd.read_csv(trainCSV, header=None)
print(trainDF)
trainFiles = trainDF[0].tolist()
trainClasses = trainDF[1].tolist()
trainFiles = trainFiles[1:]
trainClasses = trainClasses[1:]


devDF = pd.read_csv(devCSV, header=None)
print(devDF)
devFiles = devDF[0].tolist()
devClasses = devDF[1].tolist()
devFiles = devFiles[1:]
devClasses = devClasses[1:]

labels = ['A', 'B', 'C']

TargetSize = (224, 224)
def prepare_image(filepath):
    img = cv2.imread(filepath)
    # get image height, width
    # (h, w) = img.shape[:2]
    # print(h, w)
    img_resized = cv2.resize(img, TargetSize, interpolation=cv2.INTER_CUBIC)
    img_result  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_result

# plt.imshow(prepare_image(trainPath+trainFiles[1]))
# plt.show()

trainX = []
[trainX.append(prepare_image(trainPath + file)) for file in trainFiles]
# you should not apppend this to train, because of overfitting
# [trainX.append(prepare_image(devPath + file)) for file in devFiles]
trainX = np.asarray(trainX)    
# Convert Y_data from {'A','B','C'} to {0,1,2}
trainY = []
[trainY.append(ord(trainClass) - 65) for trainClass in trainClasses]
# you should not apppend this to train, because of overfitting
# [trainY.append(ord(devClass) - 65) for devClass in devClasses]

# Data Normalisation
trainX = trainX / 255.0
# One-hot encoding
trainY = to_categorical(trainY)

validX = []
[validX.append(prepare_image(devPath+file)) for file in devFiles]
validX = np.asarray(validX)    
# Convert Y_data from char to integer
validY = []
[validY.append(ord(devClass) - 65) for devClass in devClasses]

# Data Normalisation
validX = validX / 255.0
# One-hot encoding
validY = to_categorical(validY)

num_classes = 3

input_shape = trainX.shape[1:]
print(input_shape)





# Build Model 
model = Sequential()

# 1st Conv layer
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# 2nd Conv layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# 3rd Conv layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# 4th Conv layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# 5th Conv layer
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# Fully-Connected layer
model.add(Flatten())
model.add(Dense(7680, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

opt = SGD(lr=0.01)
model.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 16
num_epochs = 50

history = model.fit(trainX,trainY,batch_size=batch_size,epochs=num_epochs, validation_data=(validX,validY))