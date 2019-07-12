import os
import csv

samples = []
with open('/opt/carnd_p3/mydata/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	# Skipping the headers
	next(reader, None)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					name = '/opt/carnd_p3/mydata/IMG/'+batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					center_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					center_angle = float(batch_sample[3])
					images.append(center_image)
					
					# create adjusted steering measurements for the side camera images
					correction = 0.2 # this is a parameter to tune
					left_angle = center_angle + correction
					right_angle = center_angle - correction
					if i == 0:
						adjusted_angle = center_angle
					elif i == 1:
						adjusted_angle = left_angle
					elif i == 2:
						adjusted_angle = right_angle	
						
					angles.append(adjusted_angle)
					
			# trim image to only see section with road
			augmented_images = []
			augmented_angles = []
			for image,angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(cv2.flip(image,1))
				augmented_angles.append(angle*-1.0)
			
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
print(model.summary())

from keras.models import Model
import matplotlib.pyplot as plt
import math

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=15, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')

model.save('model.h5')