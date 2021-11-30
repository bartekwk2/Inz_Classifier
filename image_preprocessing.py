import os
import cv2
import random
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import shutil
from PIL import Image
from skimage import io

def loadImages(folderNameAll,fetchPhotos):
    imageList = []
    label = -1
    labelNames = []
    for root,_,files in os.walk(folderNameAll):

        if label >= 0 :
            folderName = os.path.basename(root)
            labelNames.append(folderName)
            print(folderName)
            images = []
            imageList.append(images)
            if fetchPhotos:
                for file in files:
                    image = cv2.imread(f"{root}/{file}")
                    image = modifyImage(image)
                    images.append((label,image))

        label+=1
    return imageList,labelNames


def modifyImage(image):
    image = cv2.resize(image,(64,64))
    return image


def divideAllImages(images,percent):
    allImagesTrain = []
    allImagesTest = []

    for imagesType in images:
        trainingSet,testSet = divideImagesByType(imagesType,percent)
        allImagesTrain.extend(trainingSet)
        allImagesTest.extend(testSet)

    random.shuffle(allImagesTrain)
    random.shuffle(allImagesTest)

    return allImagesTrain,allImagesTest


def divideImagesByType(imageTypeList,percent):
    lenType = len(imageTypeList)
    rangeType = int(lenType*percent)
    testSet = []

    for i in range(rangeType):
        indexChosen = random.randrange(lenType-i)
        valueChosen = imageTypeList[indexChosen]
        imageTypeList.pop(indexChosen)
        testSet.append(valueChosen)

    return imageTypeList,testSet


def extractLabels(imagesWithLabels,classCount):
    labels = []
    images = []

    for label,image in imagesWithLabels:
        labels.append(label)
        images.append(image)
        
    return prepareLabels(labels,classCount),prepareImages(images)


def prepareImages(images):
    size = 64
    images= np.array(images).reshape(np.array(images).shape[0],size,size,3)
    images = np.array(images).astype('float32')
    return images


def prepareLabels(label,classCount):
    label = np.array(label).astype('float32')
    label = np_utils.to_categorical(label,classCount)
    return label


def augmentImages(nameIn,nameOut,nrOfCopies):

    datagen = ImageDataGenerator(rotation_range=30,zoom_range=0.1,
                    width_shift_range=0.1,height_shift_range=0.1,
                    shear_range=0.1,horizontal_flip=True, vertical_flip=True,
                    rescale=1/255.0)

    directory = os.getcwd()
    datasetDirIn = directory + f"/{nameIn}/"
    datasetDirOut = directory + f"/{nameOut}/"

    if not os.path.exists(datasetDirOut):
        os.makedirs(datasetDirOut)

    counter = 0
    for root, _, _ in os.walk(nameIn):  
        if counter> 0 :
            pathName = os.path.basename(root)
            augmentImagesInFolder(datasetDirIn+pathName+"/",datasetDirOut+pathName+"/",nrOfCopies,datagen)
        counter+=1



def augmentImagesInFolder(image_directory, save_dir, nr_of_copies,datagen:ImageDataGenerator):

    print(save_dir)
    dataset =[]

    makeDirectory(save_dir)

    my_images = os.listdir(image_directory)
    for i, image_name in enumerate(my_images):
        if (image_name.split('.')[1]== 'jpg'):
            image = io.imread(image_directory + image_name)
            image = Image.fromarray(image)
            dataset.append(np.array(image))
            image.save(save_dir + image_name,)

    x = np.array(dataset)

    i = 0
    for j in datagen.flow(x, batch_size=len(x), save_to_dir=save_dir, save_prefix='aug', save_format='jpg'):
        if i > nr_of_copies:
            break
        i += 1


def makeDirectory(save_dir):
    if  os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)


def selectImages(image_directory,save_dir,period):

    for root, dirs, _ in os.walk(image_directory):
        for dirName in dirs:
            path = os.path.join(root, dirName)
            savePath = os.path.join(save_dir, dirName)
            makeDirectory(savePath)
            for index,fileName in enumerate(os.listdir(path)):
                if index % period == 0:
                    image = io.imread(f"{path}\{fileName}")
                    image = Image.fromarray(image)
                    image.save(f"{savePath}\{fileName}")
                

def readFromCsv(path):
    data = pd.read_csv(path)
    labels = data['label'].values
    data.drop('label',inplace=True,axis=1)
    images = data.values
    images = images.reshape(-1,28,28,1)
    labelsCount = max(labels)+1
    return prepareLabels(labels,labelsCount),images


def saveLabelNames(labels):
    with open('labels.txt','w') as f:
        for label in labels :
            f.write(label + "\n")


def getLabelNames():
    return ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]


