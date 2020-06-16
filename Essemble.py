import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CNN import CNN
import os

class Essemble:
    DATA_DIR = 'C:\\Users\\User\\Desktop\\Mango\\MangoRecongnition\\MyTrainingData'

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        dtype='float32'
    )
    test_generator = test_datagen.flow_from_directory(
        directory=DATA_DIR + '/test', # adjust here
        target_size=(224, 224),
        batch_size=16,
        shuffle=False,
    )

    models_name = ['Adadelta1', 'Adadelta2', 'Adagrad1', 'Adagrad2', 'adam1', 'adam2', 'Nadam1', 'Nadam2', 'GanAdadelta1', 'GanAdadelta2', 'GanAdadelta3']

    filenames = test_generator.filenames
    filenames = [os.path.basename(filename) for filename in filenames]
    filenames = [os.path.splitext(filename)[0] for filename in filenames]
    class_map = {idx:cls for cls, idx in list(CNN.train_generator.class_indices.items())}

    def readComparedData(self):
        # read compared data
        self.devDF = pd.read_csv('Data/dev.csv', header=None)
        self.devFiles = self.devDF[0].tolist()
        self.devClasses = self.devDF[1].tolist()
        self.devFiles = self.devFiles[1:] # remove header
        self.devClasses = self.devClasses[1:] # remove header
    
    # load models from file
    def load_all_model(self, models_name):
        all_models = list()
        for name in self.models_name:
            # load model from file
            tempModel = tf.keras.models.load_model('models/model_'+str(name)+'.h5')
            print('loading model {}...'.format(name))
            # add to list of members
            all_models.append(tempModel)
        return all_models
    
    def predict(self):
        members = self.load_all_model(self.models_name)
        print('load all model, length is {}'.format(len(members)))

        print('calculating predict result...')
        Preds = [member.predict(self.test_generator) for member in members]
        print('get index...')
        Preds_idx = [pred.argmax(axis=1) for pred in Preds]
        print('get result...')
        Preds_result = [ [self.class_map[idx] for idx in pred_idx] for pred_idx in Preds_idx ]

        # for idx, pred in enumerate(Preds):
        #     print(idx)
        #     print(pred)


        # show single accuraacy
        # for i in range(len(members)):
        #     print("accuracy {}:".format(self.models_name[int(i)]))
        #     num = 0
        #     for j in range(len(Preds_result[i])):
        #         if Preds_result[i][j] == self.devClasses[j]:
        #             num = num + 1
        #     print(num/800)

        # calculate essemble result 
        TotalPred = np.zeros(Preds[0].shape)
        for pred in Preds:
            TotalPred = TotalPred + pred
        TotalPred_idx = TotalPred.argmax(axis=1)
        TotalPred_result = [self.class_map[idx] for idx in TotalPred_idx]

        sub = pd.DataFrame({"ImageID": self.filenames, "PredictedLabel": (TotalPred_idx+1)})
        sub.to_csv("sample.csv", index = False, header = True)

        # show essemble result
        # print('TotalPred_result')
        # num = 0
        # for i in range(len(TotalPred_result)):
        #     # print(pred_result[i])
        #     if TotalPred_result[i] == self.devClasses[i]:
        #         num = num + 1
        # print(num/800)