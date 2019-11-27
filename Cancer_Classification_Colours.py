##############################################################
### Finding Average Colours of Tumour Images
##############################################################

import random
import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import csv
import seaborn as sn
import pandas as pd
import time

""" Define the parameters of the system """
Image_Size_x = 28 # Number of pixels in each image on the x-axis (original is 600)
Image_Size_y = 28 # Number of pixels in each image on the y-axis (original is 450)
coloured = 1 # 0 for grayscale, 1 for coloured
sizes = [Image_Size_x*Image_Size_y*(2*coloured+1)+3, 20, 20, 20, 7] # Layers of the neural network. First and last are fixed, can change the hidden layers
num_batches = 100 # How many batches we take from the training data
batch_size = 100 # How many images in each batch from the training data
eta_start = 0.1 # Start value of the learning rate parameter
eta_finish = 10**(-5) # End value of the learning rate parameter
Test_Size = 2500 # Size of the test set (10015 total images)
Train_Size = 10015 - Test_Size # Size of the training set (10015 total images)


""" Create annealing eta so that the learning rate parameter decreases wih each batch """
etas = []
for i in range(num_batches):
    etas.append(eta_start - (eta_start - eta_finish)*i/(num_batches - 1)) # Linearly decreasing eta


""" Function that creates a desired vector using the category of the image """
def des_vec(label):
        y = np.zeros(7)
        if label == 'nv':
            y[0] = 1.0
        elif label == 'mel':
            y[1] = 1.0
        elif label == 'bkl':
            y[2] = 1.0
        elif label == 'df':
            y[3] = 1.0
        elif label == 'akiec':
            y[4] = 1.0
        elif label == 'bcc':
            y[5] = 1.0
        elif label == 'vasc':
            y[6] = 1.0
        return y


""" Read in the tumour images and labels at the specified size """
def sex_num(sex):
    # Make the patient's sex numerical (the unknowns are set to 0.5)
    if sex == 'male':
        return 0.0
    elif sex == 'female':
        return 1.0
    else:
        return 0.5

def place_num(place):
    # Make location of the tumour numerical (unknown set to 0.5)
    if place == 'scalp':
        return 1
    elif place == 'face':
        return 2
    elif place == 'neck':
        return 3
    elif place == 'ear':
        return 4
    elif place == 'back':
        return 5
    elif place == 'chest':
        return 6
    elif place == 'hand':
        return 7
    elif place == 'acral':
        return 8
    elif place == 'genitals':
        return 9
    elif place == 'abdomen':
        return 10
    elif place == 'trunk':
        return 11
    elif place == 'upper extremity':
        return 12
    elif place == 'lower extremity':
        return 13
    elif place == 'foot':
        return 14
    else:
        return 15

def age_num(age):
    # Normalized age and set unknowns to 0.5
    if age == '':
        return 45
    else:
        return float(age)

print('Reading in the Tumour Images and Labels...')
# First read in the labels from the spreadsheet and create a dictionary of categrories keyed by the filenames.
# NOTE: I removed the top row from the metadata spreadsheet prior to running this
with open('HAM10000_metadata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    labels = {}
    for row in csv_reader:
        labels[row[1]+'.jpg'] = [row[2]]
# Now read in the images from the folders at the specified size. Make sure you change the directory
All_Pics = []
i = 0
for filename in os.listdir("D:\Python_Programming\STAT 841\HAM10000_images_part_1"):
        x = 'HAM10000_images_part_1/' + str(filename)
        image = cv2.imread(x, coloured)
        image = cv2.resize(image, (Image_Size_x, Image_Size_y))
        image = np.divide(image, 255)
        B = image[:,:,0]
        G = image[:,:,1]
        R = image[:,:,2]
        B_avg = np.average(B)
        G_avg = np.average(G)
        R_avg = np.average(R)
        # image = np.reshape(image, (Image_Size_x*(2*coloured+1)*Image_Size_y, 1))
        cat = labels[filename][0]
        cat_vec = des_vec(cat)
        All_Pics.append([image, cat_vec, cat, R_avg, G_avg, B_avg])
print('Halfway Done!')
for filename in os.listdir("D:\Python_Programming\STAT 841\HAM10000_images_part_2"):
        x = 'HAM10000_images_part_2/' + str(filename)
        image = cv2.imread(x, coloured)
        image = cv2.resize(image, (Image_Size_x, Image_Size_y))
        image = np.divide(image, 255)
        B = image[:,:,0]
        G = image[:,:,1]
        R = image[:,:,2]
        B_avg = np.average(B)
        G_avg = np.average(G)
        R_avg = np.average(R)
        # image = np.reshape(image, (Image_Size_x*(2*coloured+1)*Image_Size_y, 1))
        cat = labels[filename][0]
        cat_vec = des_vec(cat)
        All_Pics.append([image, cat_vec, cat, R_avg, G_avg, B_avg])

colour_avgs = []
for tumour in All_Pics:
    colour_avgs.append([tumour[3], tumour[4], tumour[5]])

col_avg_arr = np.array([colour_avgs[0]])
for i in range(len(colour_avgs)-1):
    col_avg_arr = np.vstack((col_avg_arr, colour_avgs[i+1]))


labels_array = []
for tumour in All_Pics:
    labels_array.append(tumour[2])
labels_array = np.array(labels_array)


dataset_colours = pd.DataFrame({'B': col_avg_arr[:, 0], 'G': col_avg_arr[:, 1], 'R': col_avg_arr[:, 2], 'Type': labels_array})
sn.pairplot(dataset_colours, hue="Type")

with open('HAM10000_metadata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = []
    for row in csv_reader:
        data.append([age_num(row[4]), place_num(row[6])])
    metadata = np.array(data)

dataset_meta = pd.DataFrame({'Age': metadata[:, 0], 'Localization': metadata[:, 1], 'Type': labels_array})
# sn.pairplot(dataset_meta, hue="Type")


# dataset_meta_nv = dataset_meta[dataset_meta['Type'] == 'nv']
# plt.figure()
# sn.distplot(dataset_meta_nv['Age'], bins=18)
# plt.title('nv')
# plt.figure()
# sn.distplot(dataset_meta_nv['Localization'], bins=14)
# plt.title('nv')
# dataset_meta_mel = dataset_meta[dataset_meta['Type'] == 'mel']
# plt.figure()
# sn.distplot(dataset_meta_mel['Age'], bins=18)
# plt.title('mel')
# plt.figure()
# sn.distplot(dataset_meta_mel['Localization'], bins=14)
# plt.title('mel')
# dataset_meta_bkl = dataset_meta[dataset_meta['Type'] == 'bkl']
# plt.figure()
# sn.distplot(dataset_meta_bkl['Age'], bins=18)
# plt.title('bkl')
# plt.figure()
# sn.distplot(dataset_meta_bkl['Localization'], bins=14)
# plt.title('bkl')
# dataset_meta_df = dataset_meta[dataset_meta['Type'] == 'df']
# plt.figure()
# sn.distplot(dataset_meta_df['Age'], bins=18)
# plt.title('df')
# plt.figure()
# sn.distplot(dataset_meta_df['Localization'], bins=14)
# plt.title('df')
# dataset_meta_akiec = dataset_meta[dataset_meta['Type'] == 'akiec']
# plt.figure()
# sn.distplot(dataset_meta_akiec['Age'], bins=18)
# plt.title('akiec')
# plt.figure()
# sn.distplot(dataset_meta_akiec['Localization'], bins=14)
# plt.title('akiec')
# dataset_meta_bcc = dataset_meta[dataset_meta['Type'] == 'bcc']
# plt.figure()
# sn.distplot(dataset_meta_bcc['Age'], bins=18)
# plt.title('bcc')
# plt.figure()
# sn.distplot(dataset_meta_bcc['Localization'], bins=14)
# plt.title('bcc')
# dataset_meta_vasc = dataset_meta[dataset_meta['Type'] == 'vasc']
# plt.figure()
# sn.distplot(dataset_meta_vasc['Age'], bins=18)
# plt.title('vasc')
# plt.figure()
# sn.distplot(dataset_meta_vasc['Localization'], bins=14)
# plt.title('vasc')






plt.show()






















# end
