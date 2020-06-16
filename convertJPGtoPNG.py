import cv2
import os

# this function is for read image,the input is directory name
def read_directory(directory_name, c):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name+c):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name+c + "/" + filename)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        filename = filename[:6]
        cv2.imwrite(directory_name + "_png" + c + '/' + filename + "png", img)

read_directory('MyTrainingData/train', '/A')
read_directory('MyTrainingData/train', '/B')
read_directory('MyTrainingData/train', '/C')