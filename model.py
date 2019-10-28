import os
import csv
import cv2
import math
import sklearn
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header_row = next(reader)
    for line in reader:
        samples.append(line)
print(samples[0])
print(samples[1])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from sklearn.utils import shuffle
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, int(batch_size/4)):
            batch_samples = samples[offset:offset+int(batch_size/4)]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = ndimage.imread(name)
                left_image = ndimage.imread('./data/IMG/'+batch_sample[1].split('/')[-1])
                right_image = ndimage.imread('./data/IMG/'+batch_sample[2].split('/')[-1])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(center_angle*-1.0) 
                images.append(left_image)
                angles.append(center_angle+0.3)
                images.append(right_image)
                angles.append(center_angle-0.5)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Cropping2D(cropping=((60,20),(0,0)),input_shape=(160,320,3)))#80*320*3
model.add(Lambda(lambda x: x/255.0-0.5)) #normalizing the data and mean centering the data

model.add(Conv2D(filters=24,kernel_size=5, strides=(2,2), padding='same', activation='relu'))#40*160*24
model.add(Conv2D(filters=36,kernel_size=5, strides=(2,2), padding='same', activation='relu'))#20*80*36
model.add(Conv2D(filters=48,kernel_size=5, strides=(2,2), padding='same', activation='relu'))#10*40*48
model.add(Conv2D(filters=64,kernel_size=3, activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(180, activation = 'relu'))
model.add(Dense(80))#, activation = 'relu'
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=6, verbose=1)

model.summary()
model.save('model.h5')


### plot the training and validation loss for each epoch
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()