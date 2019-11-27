###########################################################
### Classifying Skin Cancers Using Image Data initially
### sorted by localization
###########################################################

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
num_batches = 10 # How many batches we take from the training data
batch_size = 1000 # How many images in each batch from the training data
eta_start = 0.1 # Start value of the learning rate parameter
eta_finish = 10**(-5) # End value of the learning rate parameter
Test_Size = 2500 # Size of the test set (10015 total images)
Train_Size = 10015 - Test_Size # Size of the training set (10015 total images)

sizes_abdomen = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_acral = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_back = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_chest = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_neck = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_scalp = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_ear = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_face = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_foot = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_genital = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_hand = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_lower = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_upper = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_trunk = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]
sizes_unknown = [Image_Size_x*Image_Size_y*(2*coloured+1), 20, 20, 20, 7]

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
    elif place == 'genital':
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


print('Reading in the Tumour Images and Labels...')
# First read in the labels from the spreadsheet and create a dictionary of categrories keyed by the filenames.
# NOTE: I removed the top row from the metadata spreadsheet prior to running this
with open('HAM10000_metadata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    labels = {}
    for row in csv_reader:
        labels[row[1]+'.jpg'] = [row[6], row[2]]
# Now read in the images from the folders at the specified size. Make sure you change the directory
All_Pics = []
i = 0
for filename in os.listdir("D:\Python_Programming\STAT 841\HAM10000_images_part_1"):
        x = 'HAM10000_images_part_1/' + str(filename)
        image = cv2.imread(x, coloured)
        image = cv2.resize(image, (Image_Size_x, Image_Size_y))
        image = np.divide(image, 255)
        image = np.reshape(image, (Image_Size_x*(2*coloured+1)*Image_Size_y, 1))
        cat = labels[filename][1]
        cat_vec = des_vec(cat)
        loc = labels[filename][0]
        All_Pics.append([image, loc, cat_vec, cat])
        i = i + 1
print('Halfway Done!')
for filename in os.listdir("D:\Python_Programming\STAT 841\HAM10000_images_part_2"):
        x = 'HAM10000_images_part_2/' + str(filename)
        image = cv2.imread(x, coloured)
        image = cv2.resize(image, (Image_Size_x, Image_Size_y))
        image = np.divide(image, 255)
        image = np.reshape(image, (Image_Size_x*(2*coloured+1)*Image_Size_y, 1))
        cat = labels[filename][1]
        cat_vec = des_vec(cat)
        loc = labels[filename][0]
        All_Pics.append([image, loc, cat_vec, cat])
        i = i + 1

""" Separate the data into the training and testing sets """
# Separated randomly here
print('Separating Data into Training and Testing Sets...')
np.random.shuffle(All_Pics)
Train_Pics = All_Pics[0:len(All_Pics)-Test_Size]
Test_Pics = All_Pics[len(All_Pics)-Test_Size:len(All_Pics)]
print('Data Separation Complete')


""" Separate the training set by localizations """
Train_abdomen = []
Train_acral = []
Train_back = []
Train_chest = []
Train_neck = []
Train_scalp = []
Train_ear = []
Train_face = []
Train_foot = []
Train_genital = []
Train_hand = []
Train_lower = []
Train_upper = []
Train_trunk = []
Train_unknown = []
for pic in Train_Pics:
    place = pic[1]
    if place == 'scalp':
        Train_scalp.append(pic)
    elif place == 'face':
        Train_face.append(pic)
    elif place == 'neck':
        Train_neck.append(pic)
    elif place == 'ear':
        Train_ear.append(pic)
    elif place == 'back':
        Train_back.append(pic)
    elif place == 'chest':
        Train_chest.append(pic)
    elif place == 'hand':
        Train_hand.append(pic)
    elif place == 'acral':
        Train_acral.append(pic)
    elif place == 'genital':
        Train_genital.append(pic)
    elif place == 'abdomen':
        Train_abdomen.append(pic)
    elif place == 'trunk':
        Train_trunk.append(pic)
    elif place == 'upper extremity':
        Train_upper.append(pic)
    elif place == 'lower extremity':
        Train_lower.append(pic)
    elif place == 'foot':
        Train_foot.append(pic)
    else:
        Train_unknown.append(pic)


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
        desired = np.array(tumour[2])
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
net_abdomen = Network(sizes_abdomen)
net_acral = Network(sizes_acral)
net_back = Network(sizes_back)
net_chest = Network(sizes_chest)
net_neck = Network(sizes_neck)
net_scalp = Network(sizes_scalp)
net_ear = Network(sizes_ear)
net_face = Network(sizes_face)
net_foot = Network(sizes_foot)
net_genital = Network(sizes_genital)
net_hand = Network(sizes_hand)
net_lower = Network(sizes_lower)
net_upper = Network(sizes_upper)
net_trunk = Network(sizes_trunk)
net_unknown = Network(sizes_unknown)

def create_batch(set, size):
    set_nv, set_mel, set_bkl, set_df, set_akiec, set_bcc, set_vasc = [], [], [], [], [], [], []
    for tumour in set:
        if tumour[3] == 'nv':
            set_nv.append(tumour)
        elif tumour[3] == 'mel':
            set_mel.append(tumour)
        elif tumour[3] == 'bkl':
            set_bkl.append(tumour)
        elif tumour[3] == 'df':
            set_df.append(tumour)
        elif tumour[3] == 'akiec':
            set_akiec.append(tumour)
        elif tumour[3] == 'bcc':
            set_bcc.append(tumour)
        elif tumour[3] == 'vasc':
            set_vasc.append(tumour)
    batch = []
    total_size = len(set_nv) + len(set_mel) + len(set_bkl) + len(set_df) + len(set_akiec) + len(set_bcc) + len(set_vasc)
    if len(set_nv)/total_size > 0.05:
        new_nv = random.choices(set_nv, k=1*size)
        batch.extend(new_nv)
    if len(set_mel)/total_size > 0.05:
        new_mel = random.choices(set_mel, k=1*size)
        batch.extend(new_mel)
    if len(set_bkl)/total_size > 0.05:
        new_bkl = random.choices(set_bkl, k=1*size)
        batch.extend(new_bkl)
    if len(set_df)/total_size > 0.05:
        new_df = random.choices(set_df, k=1*size)
        batch.extend(new_df)
    if len(set_akiec)/total_size > 0.05:
        new_akiec = random.choices(set_akiec, k=1*size)
        batch.extend(new_akiec)
    if len(set_bcc)/total_size > 0.05:
        new_bcc = random.choices(set_bcc, k=1*size)
        batch.extend(new_bcc)
    if len(set_vasc)/total_size > 0.05:
        new_vasc = random.choices(set_vasc, k=1*size)
        batch.extend(new_vasc)
    np.random.shuffle(batch)
    return batch


# Train abdomens
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_abdomen
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_abdomen.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'abdomen set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Acrals
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_acral
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_acral.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'acral set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Backs
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_back
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_back.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'back set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Chests
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_chest
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_chest.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'chest set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Necks
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_neck
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_neck.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'neck set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Scalps
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_scalp
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_scalp.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'scalp set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Ears
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_ear
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_ear.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'ear set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Faces
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_face
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_face.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'face set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Feet
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_foot
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_foot.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'foot set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Genitals
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_genital
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_genital.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'genital set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Hands
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_hand
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_hand.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'hand set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Lower Extremities
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_lower
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_lower.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'lower extremity set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Upper Extremities
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_upper
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_upper.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'upper extremity set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Trunks
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_trunk
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_trunk.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'trunk set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )
# Train Unknowns
for batch in range(num_batches):
    start_time = time.time()
    batch_set = Train_unknown
    batch_set = create_batch(batch_set, batch_size)
    eta = etas[batch]
    for tumour in batch_set:
        net_unknown.point_train(tumour)
    elapsed_time = time.time() - start_time
    print('Batch:', str(batch + 1), ' | ', 'unknown set', ' | ', 'Batch Time =', "%.2f" % elapsed_time + 's' )

""" Determine train set error """
train_errors = 0
Conf_Mat_Train = np.zeros((7,7))
for tumour in Train_Pics:
    place = tumour[1]
    if place == 'scalp':
        guess = Guess(net_scalp.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'face':
        guess = Guess(net_face.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'neck':
        guess = Guess(net_neck.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'ear':
        guess = Guess(net_ear.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'back':
        guess = Guess(net_back.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'chest':
        guess = Guess(net_chest.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'hand':
        guess = Guess(net_hand.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'acral':
        guess = Guess(net_acral.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'genital':
        guess = Guess(net_genital.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'abdomen':
        guess = Guess(net_abdomen.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'trunk':
        guess = Guess(net_trunk.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'upper extremity':
        guess = Guess(net_upper.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'lower extremity':
        guess = Guess(net_lower.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    elif place == 'foot':
        guess = Guess(net_foot.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
    else:
        guess = Guess(net_unknown.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            train_errors = train_errors
        else:
            train_errors = train_errors + 1
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
    if desired == 'nv':
        column = 0
    elif desired == 'mel':
        column = 1
    elif desired == 'bkl':
        column = 2
    elif desired == 'df':
        column = 3
    elif desired == 'akiec':
        column = 4
    elif desired == 'bcc':
        column = 5
    elif desired == 'vasc':
        column = 6
    Conf_Mat_Train[row, column] = Conf_Mat_Train[row, column] + 1

train_error = train_errors/Train_Size
print('Train Error', str(train_error))

categories = np.array(['nv', 'mel', 'bkl', 'df', 'akiec', 'bcc', 'vasc'])
df_cm = pd.DataFrame(Conf_Mat_Train, index = [cat for cat in categories], columns = [cat for cat in categories])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

""" Determine test set error """
test_errors = 0
Conf_Mat_Test = np.zeros((7,7))
for tumour in Test_Pics:
    place = tumour[1]
    if place == 'scalp':
        guess = Guess(net_scalp.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'face':
        guess = Guess(net_face.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            tset_errors = test_errors + 1
    elif place == 'neck':
        guess = Guess(net_neck.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'ear':
        guess = Guess(net_ear.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'back':
        guess = Guess(net_back.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'chest':
        guess = Guess(net_chest.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'hand':
        guess = Guess(net_hand.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'acral':
        guess = Guess(net_acral.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'genital':
        guess = Guess(net_genital.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'abdomen':
        guess = Guess(net_abdomen.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'trunk':
        guess = Guess(net_trunk.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'upper extremity':
        guess = Guess(net_upper.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'lower extremity':
        guess = Guess(net_lower.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    elif place == 'foot':
        guess = Guess(net_foot.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
    else:
        guess = Guess(net_unknown.forward_prop(tumour))
        desired = tumour[3]
        if guess == desired:
            test_errors = test_errors
        else:
            test_errors = test_errors + 1
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
    if desired == 'nv':
        column = 0
    elif desired == 'mel':
        column = 1
    elif desired == 'bkl':
        column = 2
    elif desired == 'df':
        column = 3
    elif desired == 'akiec':
        column = 4
    elif desired == 'bcc':
        column = 5
    elif desired == 'vasc':
        column = 6
    Conf_Mat_Test[row, column] = Conf_Mat_Test[row, column] + 1

test_error = test_errors/Test_Size
print('Test Error', str(test_error))

df_cm = pd.DataFrame(Conf_Mat_Test, index = [cat for cat in categories], columns = [cat for cat in categories])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

plt.show()

























# end
