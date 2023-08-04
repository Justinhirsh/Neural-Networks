import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# helper functions
def show_min_max(array, i):
  random_image = array[i]
  print("min and max value in image: ", random_image.min(), random_image.max())


def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(" Digit " + str(labels[i]))
  plt.xticks([])
  plt.yticks([])
  plt.show()


img_rows, img_cols = 28, 28  

num_classes = 10 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
(train_images_backup, train_labels_backup), (test_images_backup, test_labels_backup) = mnist.load_data() 

print(train_images.shape) 
print(test_images.shape) 

train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

print(train_images[1232].shape)

accgraph = []
epochs = 100
model = Sequential()
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        accgraph.append(test_acc)


model.add(Conv2D(filters=64, kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(Flatten()) 
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',  metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=64, epochs=epochs, validation_data=(test_images, test_labels), shuffle=True, callbacks=[CustomCallback()],)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
scores = model.evaluate(test_images, test_labels, verbose=0)

model.save("C:\\Users\\Student\\Desktop\\Justin Hirsh\\NN\\Convo NN\\V2\\V2.h5")
# Add a print statement for the test accuracy
y = np.arange(epochs)
y += 1
y = y.tolist()
plt.plot(y, accgraph, '-', color='black')
plt.show()