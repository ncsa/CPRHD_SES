import numpy as np
import os
from shutil import copy
from sklearn.model_selection import train_test_split

filenames = os.listdir('PATH/TO/DATASET/')
# store the names for images and labels
img = []
lbl = []
for file in filenames: 
    if file.endswith('.jpg'):
        img.append(file)
    if file.endswith('.txt'):
        lbl.append(file)
        
        
img = np.asarray(img)
lbl = np.asarray(lbl)

# N is the total number of images in exisiting dataset
N = img.shape[0]

# split the dataset, 20% for testing
train_img, test_img = train_test_split(img, test_size=0.2)

print(train_img.shape)
print(test_img.shape)

# split images into train and test folders
for img in train_img:
    label = img.split('.jpg')[0] + '.txt'
    copy('ORIGINAL/FOLDER/' + label, 'DATA/Train/')
    
for img in test_img:
    label = img.split('.jpg')[0] + '.txt'
    copy('ORIGINAL/FOLDER/' + label, 'DATA/Test/')