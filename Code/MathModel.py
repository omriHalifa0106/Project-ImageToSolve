from keras.models import model_from_json
from imutils.contours import sort_contours
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def imageToExcersise_FromModel(path_image):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelMathSolve.h5")
    print("Loaded model from disk")

    image = cv2.imread(path_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    chars = []
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if w * h > 1200:
            # extract the character and threshold it to make the character
            # appear as white (foreground) on a black background, then
            # grab the width and height of the thresholded image
            roi = image[y:y + h, x:x + w]
            if w < h:
                wt = ((h - w) / 2) + 10
                ht = 10
            elif h < w:
                ht = ((w - h) / 2) + 10
                wt = 10
            else:
                wt = ht = 10
            wr = int(wt)
            hr = int(ht)
            chars = chars + [cv2.copyMakeBorder(roi, hr, hr, wr, wr, cv2.BORDER_CONSTANT, value=(255, 255, 255))]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()
    plt.close()
    list_exercise = ""
    for i in range(0, len(chars)):
        image = cv2.resize(chars[i], (128, 128))
        plt.imshow(image)
        image = img_to_array(image)
        image = image / 255.0
        prediction_image = np.array(image)
        prediction_image = np.expand_dims(image, axis=0)

        prediction = loaded_model.predict(prediction_image)
        value = np.argmax(prediction)
        if value == 10:
            list_exercise+= '+'
        elif value == 11 or value == 12:
            list_exercise+= '/'
        elif value == 14:
            list_exercise+='*'
        elif value == 15:
            list_exercise+='-'
        elif value == 16:
            list_exercise+='='
        elif value >= 0 and value <=9:
            list_exercise+= str(value)
        #print("Prediction is {}.".format(value))

    str_exercise = ''.join(list_exercise)
    print(str_exercise)
    return str_exercise