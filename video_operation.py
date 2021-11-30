import cv2
import numpy as np
import imutils
from keras import models
from matplotlib import pyplot as plt
from collections import Counter
import PySimpleGUI as sg

background = None


def runAverage(image, aWeight):
    global background

    if background is None:
        background = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, background, aWeight)


def segmentHand(image, threshold=15):
    global background
    diff = cv2.absdiff(background.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        return thresholded


def predictGesture(model,image,labels,textArrayAll,textArrayCurrent):
    image = preprocessImage(image)
    y_probability = model.predict(image)
    y_classes = y_probability.argmax(axis=-1)
    wordPredicted = labels[y_classes[0]]
    showGestureAsText(wordPredicted,textArrayAll,textArrayCurrent)


def showGestureAsText(text,textArrayAll:list,textArrayCurrent:list):
    textArrayCurrent.append(text)
    elementsNumber = len(textArrayCurrent)

    if elementsNumber == 60:
        wordChosen = [word for word, _ in Counter(textArrayCurrent).most_common(1)][0]
        if wordChosen == "del":
            textArrayAll.pop()
        elif wordChosen == "nothing":
            return
        elif wordChosen == "space":
            textArrayAll.append(" ")
        else:
            textArrayAll.append(wordChosen)
        
        print(f'WYBRANY ZNAK : {wordChosen}CAŁE ZDANIE : {textArrayAll}')

    elif elementsNumber == 110:
        textArrayCurrent.clear()


def preprocessImage(image):
    image = cv2.resize(image,(64,64))
    image = np.expand_dims(image,axis = 0)
    return image


def handDetection(labels,showPrediction):

    runAvarageWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 110, 350, 325, 590
    num_frames = 0
    model = models.load_model('gestureClassifier.h5')
    textArrayAll = []
    textArrayCurrent = []
    window = makeGUI()

    while(True):

        (_, frame) = camera.read()

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        frameCopy = frame.copy()

        roi = frame[top:bottom, right:left]
        roi = cv2.GaussianBlur(roi, (7, 7), 0)

        if num_frames < 30:
            runAverage(roi, runAvarageWeight)
        else:
            hand = segmentHand(roi)
            if hand is not None:

                if showPrediction:
                    predictGesture(model,roi,labels,textArrayAll,textArrayCurrent)
                    allSentence = ''.join([str(elem) for elem in textArrayAll])
                    window.read(timeout=0)
                    changeGUI(window,roi,'image')
                    window['txt'].update(allSentence)
                
                else:
                    thresholded = hand
                    maskedImage = cv2.bitwise_and(roi,roi,mask = thresholded)
                    edges = cv2.Canny(maskedImage,100,200,L2gradient = True)
                    cv2.imshow("ROI",roi)
                    cv2.imshow("Binary mask", thresholded)
                    cv2.imshow("Edges", edges)
                    cv2.imshow("Masked", maskedImage)


        cv2.rectangle(frameCopy, (left, top), (right, bottom), (0,255,0), 2)
        cv2.imshow("Video", frameCopy)
        num_frames += 1

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()



def makeGUI():
    sg.theme('Black')
    title = 'Tłumacz języka migowego'

    layout = [[sg.Text(title, size=(40, 1), justification='center', font='Helvetica 20',pad=(0,20))],
              [sg.Image(filename='', key='image'),sg.Text(key ='txt',size=(20, 4), justification='center', font='Helvetica 20',pad=(15,5))],
              ]

    window = sg.Window(title,layout, location=(800, 400))
    return window

def changeGUI(window,frame,name):
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window[name].update(data=imgbytes)







    