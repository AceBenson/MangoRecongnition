import os
import csv
import string
import shutil

# ----------Create a new folder for training data (classes)----------
oldFolderPath = os.getcwd()+'/Data'
newFolderPath = os.getcwd()+'/MyTrainingData'
if os.path.isdir(newFolderPath):
    print('The new folder has already exsited!')
else:
    print('Creating the new folder...')
    os.mkdir(newFolderPath)
    os.mkdir(newFolderPath + "/train")
    os.mkdir(newFolderPath + "/dev")
    os.mkdir(newFolderPath + "/train/A")
    os.mkdir(newFolderPath + "/train/B")
    os.mkdir(newFolderPath + "/train/C")
    os.mkdir(newFolderPath + "/dev/A")
    os.mkdir(newFolderPath + "/dev/B")
    os.mkdir(newFolderPath + "/dev/C")

    print('Copying the train files...')
    with open('Data/train.csv', newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            # print(row['ID'], row['Class'])
            # print(oldFolderPath + '/train/' + row['image_id'])
            shutil.copy(oldFolderPath + '/train/' + row['image_id'], newFolderPath + '/train/' + row['label'])
    print('Copying the dev files...')
    with open('Data/dev.csv', newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            # print(row['ID'], row['Class'])
            shutil.copy(oldFolderPath + '/dev/' + row['image_id'], newFolderPath + '/dev/' + row['label'])

    print('Finished!')