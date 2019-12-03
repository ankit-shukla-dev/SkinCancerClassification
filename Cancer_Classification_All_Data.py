#####################################################################
### Classifying Skin Cancers Using Image Data and Demographic Data
#####################################################################

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
batch_size = 1000 # How many images in each batch from the training data
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
        return 1/14
    elif place == 'face':
        return 2/14
    elif place == 'neck':
        return 3/14
    elif place == 'ear':
        return 4/14
    elif place == 'back':
        return 5/14
    elif place == 'chest':
        return 6/14
    elif place == 'hand':
        return 7/14
    elif place == 'acral':
        return 8/14
    elif place == 'genitals':
        return 9/14
    elif place == 'abdomen':
        return 10/14
    elif place == 'trunk':
        return 11/14
    elif place == 'upper extremity':
        return 12/14
    elif place == 'lower extremity':
        return 13/14
    elif place == 'foot':
        return 14/14
    else:
        return 0.5

def age_num(age):
    # Normalized age and set unknowns to 0.5
    if age == '':
        return 0.5
    else:
        return float(age)/85.0

print('Reading in the Tumour Images and Labels...')
# First read in the labels from the spreadsheet and create a dictionary of categrories keyed by the filenames.
# NOTE: I removed the top row from the metadata spreadsheet prior to running this
with open('HAM10000_metadata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    labels = {}
    for row in csv_reader:
        labels[row[1]+'.jpg'] = [row[2], age_num(row[4]), sex_num(row[5]), place_num(row[6])]
# Now read in the images from the folders at the specified size. Make sure you change the directory
All_Pics = []
i = 0
for filename in os.listdir("D:\Python_Programming\STAT 841\HAM10000_images_part_1"):
        x = 'HAM10000_images_part_1/' + str(filename)
        image = cv2.imread(x, coloured)
        image = cv2.resize(image, (Image_Size_x, Image_Size_y))
        image = np.divide(image, 255)
        image = np.append( np.reshape(image, (Image_Size_x*(2*coloured+1)*Image_Size_y, 1)),
                           [labels[filename][1], labels[filename][2], labels[filename][3]] )
        cat = labels[filename][0]
        cat_vec = des_vec(cat)
        All_Pics.append([image, cat_vec, cat])
        i = i + 1
print('Halfway Done!')
for filename in os.listdir("D:\Python_Programming\STAT 841\HAM10000_images_part_2"):
        x = 'HAM10000_images_part_2/' + str(filename)
        image = cv2.imread(x, coloured)
        image = cv2.resize(image, (Image_Size_x, Image_Size_y))
        image = np.divide(image, 255)
        image = np.append( np.reshape(image, (Image_Size_x*(2*coloured+1)*Image_Size_y, 1)),
                           [labels[filename][1], labels[filename][2], labels[filename][3]] )
        cat = labels[filename][0]
        cat_vec = des_vec(cat)
        All_Pics.append([image, cat_vec, cat])
        i = i + 1

""" Separate the data into the training and testing sets """
# Separated randomly here
print('Separating Data into Training and Testing Sets...')
np.random.shuffle(All_Pics)
Train_Pics = All_Pics[0:len(All_Pics)-Test_Size]
Test_Pics = All_Pics[len(All_Pics)-Test_Size:len(All_Pics)]
print('Data Separation Complete')


""" Define the activation function """
# I chose a sigmoid function for phi
def Phi(x):
    y = 1.0/(1.0 + np.exp(-x))
    return y
def dPhi(x):
    y = np.exp(-x)/(1.0 + np.exp(-x))**2
    return y


""" Define the neural network class to train on the images """
class Network(object):

    def __init__(self, sizes):
        """This defines the attributes of the network object"""
        self.num_layers = len(sizes)
        self.layer_sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward_prop(self, tumour):
        """Performs the forward propogation of the network for the input image"""
        x = np.array([pixel for pixel in tumour[0]])
        for b, w in zip(self.biases, self.weights):
            x = Phi(np.matmul(w, x).reshape(b.shape) + b)
        return x

    def point_train(self, tumour):
        """Performs the backward propogation for the input image"""
        # Initialize and compute all vs and ys through forward propogation
        b_change = [np.zeros(b.shape) for b in self.biases]
        w_change = [np.zeros(w.shape) for w in self.weights]
        desired = np.array(tumour[1])
        y = np.array([pixel for pixel in tumour[0]])
        y_layer = [y]
        v_layer = []
        for b, w in zip(self.biases, self.weights):
            v = np.matmul(w, y).reshape(b.shape) + b
            v_layer.append(v)
            y = Phi(v)
            y_layer.append(y)
        # Perform updates using back propogation
        delta_layer = np.array([(d - y)*dPhi(v) for d, y, v in zip(desired, y_layer[-1], v_layer[-1])])
        b_change[-1] = delta_layer
        w_change[-1] = np.outer(delta_layer, y_layer[-2])
        for i in range(2, self.num_layers):
            delta_layer = np.array([np.dot(w_row, delta_layer)*dPhi(v) for w_row, v in zip(np.transpose(self.weights[-i+1]), v_layer[-i])])
            b_change[-i] = delta_layer[-i]
            w_change[-i] = np.outer(delta_layer, y_layer[-i-1])
        self.weights = [w + eta*change for w, change in zip(self.weights, w_change)]
        self.biases = [b + eta*change for b, change in zip(self.biases, b_change)]


""" Function that guesses the category from the output vector """
def Guess(output):
    num = np.argmax(output)
    if num == 0:
        guess = 'nv'
    elif num == 1:
        guess = 'mel'
    elif num == 2:
        guess = 'bkl'
    elif num == 3:
        guess = 'df'
    elif num == 4:
        guess = 'akiec'
    elif num == 5:
        guess = 'bcc'
    elif num == 6:
        guess = 'vasc'
    return(guess)


""" Create and train the network using the batches from the training set """
# Randomly sample from training set with equal incidence of categories
def create_batch(set, size):
    set_nv, set_mel, set_bkl, set_df, set_akiec, set_bcc, set_vasc = [], [], [], [], [], [], []
    for tumour in set:
        if tumour[2] == 'nv':
            set_nv.append(tumour)
        elif tumour[2] == 'mel':
            set_mel.append(tumour)
        elif tumour[2] == 'bkl':
            set_bkl.append(tumour)
        elif tumour[2] == 'df':
            set_df.append(tumour)
        elif tumour[2] == 'akiec':
            set_akiec.append(tumour)
        elif tumour[2] == 'bcc':
            set_bcc.append(tumour)
        elif tumour[2] == 'vasc':
            set_vasc.append(tumour)
    new_nv = random.choices(set_nv, k=1*size)
    new_mel = random.choices(set_mel, k=1*size)
    new_bkl = random.choices(set_bkl, k=1*size)
    new_df = random.choices(set_df, k=1*size)
    new_akiec = random.choices(set_akiec, k=1*size)
    new_bcc = random.choices(set_bcc, k=1*size)
    new_vasc = random.choices(set_vasc, k=1*size)
    batch = []
    batch.extend(new_nv)
    batch.extend(new_mel)
    batch.extend(new_bkl)
    batch.extend(new_df)
    batch.extend(new_akiec)
    batch.extend(new_bcc)
    batch.extend(new_vasc)
    np.random.shuffle(batch)
    return batch

net = Network(sizes)
for batch in range(num_batches):
    start_time = time.time()
    batch_set = create_batch(Train_Pics, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net.point_train(tumour)

    # # Determine the train error (full train set, not just the batch)
    # sum_train = 0
    # for tumour in Train_Pics:
    #     guess = Guess(net.forward_prop(tumour))
    #     if guess == tumour[2]:
    #         sum_train = sum_train
    #     else:
    #         sum_train = sum_train + 1
    #
    # # Determine the test error (full test set)
    # sum_test = 0
    # for tumour in Test_Pics:
    #     guess = Guess(net.forward_prop(tumour))
    #     if guess == tumour[2]:
    #         sum_test = sum_test
    #     else:
    #         sum_test = sum_test + 1

    elapsed_time = time.time() - start_time

    # print('Batch:', str(batch + 1), ' | ', 'Train Error =', str(sum_train/len(Train_Pics)), ' | ', 'Test Error =', str(sum_test/len(Test_Pics)), ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
    print('Batch:', str(batch + 1), ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )


# Determine final train error (full train set, not just the batch)
sum_train = 0
for tumour in Train_Pics:
    guess = Guess(net.forward_prop(tumour))
    if guess == tumour[2]:
        sum_train = sum_train
    else:
        sum_train = sum_train + 1

# Determine final test error (full test set)
sum_test = 0
for tumour in Test_Pics:
    guess = Guess(net.forward_prop(tumour))
    if guess == tumour[2]:
        sum_test = sum_test
    else:
        sum_test = sum_test + 1

elapsed_time = time.time() - start_time

print('Train Error =', str(sum_train/len(Train_Pics)), ' | ', 'Test Error =', str(sum_test/len(Test_Pics)))

""" Create and Print Confusion Matrices """
categories = np.array(['nv', 'mel', 'bkl', 'df', 'akiec', 'bcc', 'vasc'])

Conf_Mat_Train = np.zeros((7,7))
for tumour in Train_Pics:
    guess = Guess(net.forward_prop(tumour))
    if guess == 'nv':
        row = 0
    elif guess == 'mel':
        row = 1
    elif guess == 'bkl':
        row = 2
    elif guess == 'df':
        row = 3
    elif guess == 'akiec':
        row = 4
    elif guess == 'bcc':
        row = 5
    elif guess == 'vasc':
        row = 6
    correct = tumour[2]
    if correct == 'nv':
        column = 0
    elif correct == 'mel':
        column = 1
    elif correct == 'bkl':
        column = 2
    elif correct == 'df':
        column = 3
    elif correct == 'akiec':
        column = 4
    elif correct == 'bcc':
        column = 5
    elif correct == 'vasc':
        column = 6
    Conf_Mat_Train[row, column] = Conf_Mat_Train[row, column] + 1

Conf_Mat_Test = np.zeros((7,7))
for tumour in Test_Pics:
    guess = Guess(net.forward_prop(tumour))
    if guess == 'nv':
        row = 0
    elif guess == 'mel':
        row = 1
    elif guess == 'bkl':
        row = 2
    elif guess == 'df':
        row = 3
    elif guess == 'akiec':
        row = 4
    elif guess == 'bcc':
        row = 5
    elif guess == 'vasc':
        row = 6
    correct = tumour[2]
    if correct == 'nv':
        column = 0
    elif correct == 'mel':
        column = 1
    elif correct == 'bkl':
        column = 2
    elif correct == 'df':
        column = 3
    elif correct == 'akiec':
        column = 4
    elif correct == 'bcc':
        column = 5
    elif correct == 'vasc':
        column = 6
    Conf_Mat_Test[row, column] = Conf_Mat_Test[row, column] + 1

df_cm = pd.DataFrame(Conf_Mat_Train, index = [cat for cat in categories], columns = [cat for cat in categories])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
df_cm = pd.DataFrame(Conf_Mat_Test, index = [cat for cat in categories], columns = [cat for cat in categories])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

















# End of Code
