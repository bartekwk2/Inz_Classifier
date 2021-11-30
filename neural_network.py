from keras import Sequential
import tensorflow as tf
from keras.callbacks.callbacks import History
from keras.layers import Conv2D,Activation,MaxPooling2D,Conv2D,Flatten,Dense
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import itertools
import numpy as np
import pandas as pd


def createModel():

    size = 64
    dim = 3
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(size, size, dim)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(29, activation='softmax'))

    model.summary()
    return model


def learnNetwork(model:Sequential,train,label):

    model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    history = model.fit(train, label, 
                    validation_split=0.03,
                    shuffle=True,
                    batch_size=32, epochs=10)

    displayTrainStatistics(history)


def testNetwork(model:Sequential,test,label_test,labelNames):
    test_loss, test_acc = model.evaluate(test, label_test)
    print('Test accuracy:', test_acc)
    predictions = np.round(model.predict(test),0)
    displayTestStatistics(label_test,predictions,labelNames)


def displayTestStatistics(label_test,preds,labelNames):
    categorical_test_labels = pd.DataFrame(label_test).idxmax(axis=1)
    categorical_predictions = pd.DataFrame(preds).idxmax(axis=1)
    my_confussion_matrix = confusion_matrix(categorical_test_labels,categorical_predictions)
    plotConfusionMatrix(my_confussion_matrix,labelNames)
    classification_metrics = metrics.classification_report(label_test, preds, target_names=labelNames,zero_division= False)
    print(classification_metrics)



def plotConfusionMatrix(matrix, classes,
             normalize=False,
             title='Macierz pomyłek',
             cmap=plt.cm.get_cmap('Greens',10)):

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

       
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def displayTrainStatistics(history:History):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val'], loc='upper left')
    plt.show()


def tfLiteConvertion(modelName):
    model = tf.keras.models.load_model(modelName +'.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(f"mobile{modelName}.tflite", "wb").write(tflite_model)




    

