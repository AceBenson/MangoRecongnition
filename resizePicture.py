import os
import pandas as pd
import numpy as np
import cv2


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
    img_resized = cv2.resize(img, TargetSize, interpolation=cv2.INTER_CUBIC)
    return img_resized

for file in trainFiles:
    new_image = prepare_image(trainPath+file)
    cv2.imwrite('Data/train_resized/'+file, new_image)

for file in devFiles:
    new_image = prepare_image(devPath+file)
    cv2.imwrite('Data/dev_resized/'+file, new_image)
