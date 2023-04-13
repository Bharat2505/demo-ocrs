#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
import random 
import cv2
import imutils
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
LB = LabelBinarizer()
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import pickle
from flask import Flask, render_template, request
from keras.models import model_from_json
from werkzeug.utils import secure_filename


# In[ ]:




json_file = open('own_u_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#### load weights into new model
loaded_model.load_weights("own_u_model.h5")
print("Loaded model from disk")


app = Flask(__name__,template_folder = 'templates')



@app.route('/')
def home():
    return render_template('index.html')

pick = open('labels_uyir.pickle','rb')
y= pickle.load(pick)
pick.close()
dat = list(y.keys())

new_val_y=[]
for labels in dat: #feature,
   
    new_val_y.append(labels)
new_val_y = np.array(new_val_y)
new_val_y = LB.fit_transform(new_val_y)
def get_letters(img):
    letters = []
    image = cv2.imread(img)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = Image.open(img)
    #image = np.asarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = cv2.merge([thresh, thresh, thresh])
        thresh = thresh.reshape(1,32,32,3)
        ypred = loaded_model.predict(thresh)
        ypred = LB.inverse_transform(ypred)
        #result = np.argmax(ypred)
        #ypred = determine_character(result)
        [x] = ypred
        letters.append(x)
    return letters, image


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def get_word(letter):
    word = "".join(letter)
    return word
path = 'inputs/'
@app.route('/upload',methods=['POST','GET'])
def upload():
    image = request.files['image']
    filename = secure_filename(image.filename)
    image.save(os.path.join(path,filename))
    img_path = os.path.join(path,filename)
    letter,image = get_letters(img_path)
    word = get_word(letter)
    return render_template('result.html',text=word)

if __name__ == '__main__':
    app.run(debug=True)
    

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




