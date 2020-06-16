from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import tensorflow as tf

class CNN:
    DATA_DIR = 'C:\\Users\\User\\Desktop\\Mango\\MangoRecongnition\\DataAddGan'

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
        rescale=1./255,
        dtype='float32'
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

    # take dev as test data, and validation is split from train
    # use real "test" data when 6/17
    # test_generator = test_datagen.flow_from_directory(
    #     directory=DATA_DIR + '/origin_dev',
    #     target_size=(224, 224),
    #     batch_size=16,
    #     shuffle=False,
    # )


    # ----------Build model----------
    def createModel(self):
        print('Createing Model...')
        self.model = Sequential()
        # 1st Conv layer
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # 2nd Conv layer
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # 3rd Conv layer
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # 4th Conv layer
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # 5th Conv layer
        self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # Fully-Connected layer
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        
        opt = 'adam'
        if self.name == 'Nadam':
            print("Use Nadam")
            opt = optimizers.Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        elif self.name == 'Adadelta':
            print("Use Adadelta")
            opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        elif self.name == 'Adagrad':
            print("Use Adagrad")
            opt = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        elif self.name == 'GanAdadelta':
            print("Use GanAdadelta")
            opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.summary()

        print('End of creating model')

    def trainingModel(self):
        # adjust optimizer 
        m_callbacks = tf.keras.callbacks.ModelCheckpoint('./models/model_' + self.name +  str(self.index) + '.h5', save_best_only=True, save_weights_only=False)
        self.History = self.model.fit(self.train_generator, callbacks=[m_callbacks], epochs=30, validation_data=self.validation_generator)
        # self.model.save('models/model_' + self.name + str(self.index) + '.h5')

    def loadModel(self):
        self.model = tf.keras.models.load_model('models/model_' + self.name + str(self.index) + '.h5')
        return self.model

    def showFigure(self):
        plt.figure(figsize = (15,5))
        plt.subplot(1,2,1)
        plt.plot(self.History.history['accuracy'])
        plt.plot(self.History.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(1,2,2)
        plt.plot(self.History.history['loss'])
        plt.plot(self.History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('pictures/'+self.name+'_accandloss_'+str(self.index)+'.png')
        # plt.show()
        # plt.show()

    def predict(self):
        self.pred = self.model.predict(self.test_generator)

        filenames = self.test_generator.filenames
        class_map = {idx:cls for cls, idx in list(self.train_generator.class_indices.items())} 
        pred_idx = self.pred.argmax(axis=1)
        pred_result = [class_map[idx] for idx in pred_idx]

        sub = pd.DataFrame({"image_id": filenames, "label": pred_result})
        sub.to_csv("result_" + self.name + ".csv", index = False, header = True)

    def __init__(self, name, index):
        print("init")
        self.name = name
        self.index = index