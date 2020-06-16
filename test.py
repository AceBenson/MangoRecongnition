from CNN import CNN

# models_name = ['Adadelta', 'Adagrad', 'adam', 'Nadam']

# for name in models_name:
#     for i in range(1, 3):
#         CNNmodel = CNN(name, i)
#         CNNmodel.createModel()
#         CNNmodel.trainingModel()
#         CNNmodel.showFigure()

for i in range(1, 4):
    CNNmodel = CNN('GanAdadelta', i)
    CNNmodel.createModel()
    CNNmodel.trainingModel()
    CNNmodel.showFigure()
