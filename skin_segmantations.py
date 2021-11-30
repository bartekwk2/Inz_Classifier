import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def loadImages():
    images = []
    for image_name in imageNames:
        image_path = os.path.join(path, image_name)
        images.append(cv2.imread(image_path))
    return images

def detectionEdges(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(image,(5,5),0)
    edges=cv2.Canny(gray,30,60,L2gradient = True)
    return edges

def detectionTreshhold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (41, 41), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = np.array(thresh)
    thresh = 255 - thresh
    return thresh

def detectionHsv(bgr_image):
    hsv_lower = np.array([0, 48, 80], dtype="uint8")
    hsv_upper = np.array([20, 255, 255], dtype="uint8")
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    skin_region = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
    #blurred = cv2.blur(skin_region, (2,2))
    #_,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    return skin_region


def detectionHsv2(bgr_image):
    hsv_lower_2 = np.array([0, 50, 0], dtype="uint8")
    hsv_upper_2 = np.array([120, 150, 255], dtype="uint8")
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    skin_region = cv2.inRange(hsv_image, hsv_lower_2, hsv_upper_2)
    return skin_region


def detectionYcrcb(bgr_image):
    # Wartości wzięte z publikacji: 'Face Segmentation Using Skin-Color Map in Videophone Applications'
    lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
    upper_ycrcb = np.array([255, 173, 127], dtype="uint8")
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
    skin_region = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
    return skin_region


def isSkin(b,g,r):
    # Wartości wzięte z publikacji: 'RGB-H-CbCr Skin Colour Model for Human Face Detection'
    e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
    abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
    return e1 or e2

def detectionBgr(bgr_image):
    height = bgr_image.shape[0]
    width = bgr_image.shape[1]
    image = np.zeros((height, width, 1), dtype="uint8")

    # Ustawienie tylko pikseli należących do skóry na kolor biały
    for y in range(height):
        for x in range(width):
            (b, g, r) = bgr_image[y, x]
            if isSkin(b, g, r):
                image[y, x] = 255
    return image


def detectionMediaPipe(image):

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        #image = cv2.flip(image,0)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print('HAND:', results.multi_handedness)
        width,height,_ = image.shape
        coords_found = []
        pixels_found = []

        pixels_found_R = []
        pixels_found_G = []
        pixels_found_B = []

        if  results.multi_hand_landmarks:
            annotated_image = image.copy()
            hand_land_mark_all = results.multi_hand_landmarks[0]
            '''
            mp_drawing.draw_landmarks(
                annotated_image, hand_land_mark_all, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("DETECTED",cv2.flip(annotated_image, 1))
            cv2.waitKey(0)
            '''
            for hand_land_mark in hand_land_mark_all.landmark:
                x_value = round(hand_land_mark.x * width)
                y_value = round(hand_land_mark.y * height)
                pixels_found.append(image[y_value,x_value])
                pixels_found_R.append(image[y_value,x_value][2])
                pixels_found_G.append(image[y_value,x_value][1])
                pixels_found_B.append(image[y_value,x_value][0])
                coords_found.append((y_value,x_value))
           
            avg_R = np.average(pixels_found_R)
            avg_G = np.average(pixels_found_G)
            avg_B = np.average(pixels_found_B)

            treshhold = 25

            lower = np.array([checkBoundry(True,avg_B-treshhold), checkBoundry(True,avg_G-treshhold), checkBoundry(True,avg_R-treshhold)], dtype="uint8")
            upper = np.array([checkBoundry(False,avg_B+treshhold), checkBoundry(False,avg_G+treshhold), checkBoundry(False,avg_R+treshhold)], dtype="uint8")
            skin_region = cv2.inRange(image, lower, upper)

            return skin_region


def applyDetectors(image):
    skinDetectors = [detectionHsv, detectionHsv2, detectionYcrcb, detectionBgr,detectionTreshhold, detectionEdges,detectionMediaPipe]
    skinImages = [image]

    for detector in skinDetectors:
        try:
            skinDetected = detector(image)
            result = cv2.cvtColor(skinDetected, cv2.COLOR_GRAY2BGR)
            skinImages.append(result)
        except:
            skinImages.append(errorImage)

    return skinImages


def showAllImages(images):
    indexAll = 1
    showLabels = True
    for image in images:
        skinImages = applyDetectors(image)
        showImages(skinImages, indexAll,showLabels)
        showLabels = False
        indexAll += len(skinImages)
    plt.show()


def showImages(images, pos,showLabels):
    detectorsNames = ["Original","Hsv","Hsv2","Ycrcb","Bgr","Gray+Treshhold","Gray+Edges","MediaPipe"]
    for index, image in enumerate(images):
        # Konwersja z przestrzeni BGR do RGB
        img_RGB = image[:, :, ::-1]
        newTitle = detectorsNames[index]
        newPos = pos + index
        plt.subplot(len(imageNames), len(detectorsNames), newPos)
        plt.imshow(img_RGB)
        plt.axis('off')
        if showLabels :
            plt.title(newTitle)


def checkBoundry(lower,value):
    if lower:
        if value <= 0 :
            return 0
        else:
            return value
    else:
        if value >= 255 :
            return 255
        else:
            return value

        



# path = 'data\zdjecia_proba'
# imageNames = ['A\A1.jpg','A\A505.jpg']
# errorImage = cv2.imread(os.path.join(path, 'error.jpg'))
# plt.figure(figsize=(15, 8))
# plt.suptitle("Segmentacja skóry przy użyciu różnych przestrzeni barw", fontsize=14, fontweight='bold')
# images = loadImages()
# showAllImages(images)



def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)


img = cv2.imread('data\zdjecia_proba\A\A1.jpg')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(img_yuv)

lut_u, lut_v = make_lut_u(), make_lut_v()

# Convert back to BGR so we can apply the LUT and stack the images
y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

u_mapped = cv2.LUT(u, lut_u)
v_mapped = cv2.LUT(v, lut_v)

result = np.vstack([img, y, u_mapped, v_mapped])

cv2.imwrite('shed_combo.png', result)
