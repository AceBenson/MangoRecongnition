import cv2
import os

# this function is for read image,the input is directory name
def read_directory(directory_name, destination):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r''+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        filename = filename[:-4]
        # print(filename)
        # cv2.imwrite(directory_name + "_png" + c + '/' + filename + "png", img)
        cv2.imwrite(destination + filename + '.jpg', img)

read_directory('C:\\Users\\User\\Desktop\\picture-gen-C\\generated_224x224', 'C:\\Users\\User\\Desktop\\Mango\\MangoRecongnition\\DataAddGan\\GanC\\')
# read_directory('MyTrainingData/train', '/B')
# read_directory('MyTrainingData/train', '/C')