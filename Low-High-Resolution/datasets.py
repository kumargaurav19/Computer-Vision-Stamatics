#!/usr/bin/env python
# coding: utf-8

# In[87]:


import os, re
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm


# In[88]:


def sorted_files(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9])+',key)]
    return sorted(data , key=alphanum_key)


# In[89]:


def load(path):
    files = os.listdir(path)
    files = sorted_files(files)
    image = []
    for i in tqdm(files):        
        img = cv2.imread(path + '/'+i,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        image.append(img_to_array(img))
    return image


# In[90]:


def load_dataset():
    path1 = 'image low-high\\dataset\\train\\low_res'
    path2 = 'image low-high\\dataset\\train\\high_res'
    path3 = 'image low-high\\dataset\\val\\low_res'
    path4 = 'image low-high\\dataset\\val\\high_res'
    train_low = load(path1)
    train_high = load(path2)
    test_low = load(path3)
    test_high = load(path4)
    return train_low,train_high,test_low,test_high


# In[ ]:




