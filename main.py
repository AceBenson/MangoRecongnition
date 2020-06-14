from CNN import CNN

if __name__ == '__main__':
    m_CNN = CNN()

    m_CNN.createModel()
    m_CNN.trainingModel()
    m_CNN.showFigure()
    # m_CNN.loadModel('MyModel')

    m_CNN.predict()
