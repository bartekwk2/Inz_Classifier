import cv2
import numpy as np
from video_operation import handDetection
from image_preprocessing import loadImages,divideAllImages,extractLabels,prepareLabels,readFromCsv,getLabelNames,augmentImages,selectImages,saveLabelNames
from neural_network import createModel,learnNetwork,testNetwork,tfLiteConvertion


def makeClassifier():
    # Preprocessing zdjęć
    allImages,labelNames = loadImages("data/jeden/zdjecia_aug",True)

    imagesTrain,imagesTest = divideAllImages(allImages,0.25)
    classCount = len(labelNames)

    trainLabels,trainImages = extractLabels(imagesTrain,classCount)
    testLabels,testImages = extractLabels(imagesTest,classCount)

    # Tworzenie modelu sieci
    model = createModel()
    learnNetwork(model,trainImages,trainLabels)
    testNetwork(model,testImages,testLabels,labelNames)
    model.save('gestureClassifier.h5')


def secondDataSet():
    trainLabels,trainImages= readFromCsv('data/zdjecia_dwa/sign_mnist_train.csv')
    testLabels,testImages= readFromCsv('data/zdjecia_dwa/sign_mnist_test.csv')
    labelNames = getLabelNames()



#tfLiteConvertion('gestureClassifier')
#selectImages('data/jeden/zdjecia','data/jeden/zdjecia_sel',4)
#augmentImages('data/jeden/zdjecia_sel','data/jeden/zdjecia_aug',4)
#makeClassifier()



# Detekcja wzorca z filmu
_,labelNames = loadImages("data/jeden/zdjecia_aug",False)

handDetection(labelNames,True)


