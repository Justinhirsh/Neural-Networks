import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense, Flatten

# helper functions
def show_min_max(array, i):
  random_image = array[i]
  print(random_image.min(), random_image.max())

def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(" Digit " + str(labels[i]))
  plt.xticks([])
  plt.yticks([])
  plt.show()

img_rows = 28
img_cols = 28
num_classes = 10
input_shape = (28,28,1)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
(train_images_backup, train_labels_backup), (test_images_backup, test_labels_backup) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
# print(f"Train Shape = {train_images.shape}")
# print(f"Test Shape = {test_images.shape}")
# plot_image(train_images, 100, train_labels)
# out = ""
# print("Raw out")
# for i in range(28):
#   for j in range(28):
#     f = int(train_images[100][i][j][0])
#     s = "{:3d}".format(f)
#     out += (str(s)+" ")
#   print(out)
#   out = ""
# print("Min max train images")
# show_min_max(train_images, 100)
train_images = train_images.astype('float32')
# Add the code for test images here:
test_images = test_images.astype('float32')
train_images /= 255 
# Add the code for test images here:
test_images /= 255
# plot_image(train_images, 100, train_labels)

# out = ""
# for i in range(28):
#   for j in range(28):
#     f = (train_images[100][i][j][0])
#     s = "{:0.1f}".format(f)
#     out += (str(s)+" ")
# #   print(out)
#   out = ""

# print("Simple out")
# show_min_max(train_images, 100)
train_labels = keras.utils.to_categorical(train_labels, num_classes) 
# Add the code for test images here:
test_labels = keras.utils.to_categorical(test_labels, num_classes)
epochs = 100
model = Sequential()
model.add(Flatten(input_shape= input_shape))

model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
# model.add(Dense(units=1000, activation='gelu'))
# model.add(Dense(units=500, activation='gelu'))
# model.add(Dense(units=500, activation='gelu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=epochs, shuffle=True)
model.save('impmodel.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# Add a print statement for the test accuracy
print('\nTest accuracy:', test_acc)