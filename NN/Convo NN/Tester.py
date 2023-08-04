from keras.preprocessing import image
from PIL import Image,ImageChops 
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2

version = 'V2'
# Load the image
img = cv2.imread("C:\\Users\\Student\\Desktop\\Justin Hirsh\\NN\\Convo NN\\test0.png")

# Convert the image to an array
arr = np.array(img)
arr = arr[:,:,0]
arr = arr.reshape(28,28,1)
# Print the shape of the array
print(arr.shape)

# Load in the original data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("MNIST data loaded")

# This will work for the models if you download them from the links above. 
# If you want to export your own models, use the name of them here instead. 
my_model = tf.keras.models.load_model(f"C:\\Users\\Student\\Desktop\\Justin Hirsh\\NN\\Convo NN\\{version}\\{version}.h5")


# A function to graph an image with the label
def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(" Digit " + str(labels[i]))
  plt.xticks([])
  plt.yticks([])
  plt.show()

# A function to use the model and image array (x) to make a prediction. The image will also be shown, and 
# labelled with the predicted label. 
def predict_image(model, x):
  x = x.astype('float32')
  x = x / 255.0
  x = x.reshape(28,28,1)
  x = np.expand_dims(x, axis=0)

  image_predict = model.predict(x, verbose=0)
  print("Predicted Label: ", np.argmax(image_predict))

  plt.imshow(np.squeeze(x))
  plt.xticks([])
  plt.yticks([])
  plt.show()
 
  # uncomment this like if you want to see the array of predictions
  # print(image_predict)
  return image_predict

# Create a bar plot of the predictions array. The true label will be marked in blue, and the predicted label 
# (if different than the true label) will be marked red. The h argument dictates how tall the barplot is. 
def plot_value_array(predictions_array, true_label, h):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array[0], color="#777777")
  plt.ylim([(-1*h), h])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  plt.show()
predict_image(my_model, arr)