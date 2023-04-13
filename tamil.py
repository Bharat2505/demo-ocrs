#!/usr/bin/env python
# coding: utf-8

# In[3]:


##importing the necessary packages


import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model,Sequential
from keras.layers import Input,InputLayer, Conv2D, MaxPooling2D, Reshape, Bidirectional, GRU, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.models import load_model 
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# In[4]:


def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


# In[5]:


#alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
alphabets ='அஆஇஈஉஊஎஏஐஒஓஔஃகசஙஞடணதநபமயரலளறவழன-'
max_str_len = 10 # max length of input labels
num_of_characters = len(alphabets) +1 # +1 for ctc pseudo blank
num_of_timestamps = 3 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret


# In[10]:

app = Flask(__name__,template_folder = 'templates')



@app.route('/')

def home():
    return render_template('index.html')


def prediction(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(image, cmap='gray')
    model = load_model("basemodel-gru1.h5")
    image = preprocess(image)
    image = image/255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                   greedy=True)[0][0])
#plt.title(num_to_label(decoded[0]), fontsize=12)
    l = []
    for i in num_to_label(decoded[0]):
        l.append(i)
    
    output = l[0]
    return output
path = 'inputs/'
@app.route('/upload',methods=['POST','GET'])
def upload():
    image = request.files['image']
    filename = secure_filename(image.filename)
    image.save(os.path.join(path,filename))
    img_path = os.path.join(path,filename)
    letter = prediction(img_path)
    return render_template('result.html',text=letter)


if __name__ == '__main__':
    app.run(debug=True)
    

# In[11]:




# In[ ]:




