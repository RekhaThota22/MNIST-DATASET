# MNIST-DATASET
# Importing Libraries:
import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras import layers,models,callbacks
import matplotlib.pyplot as plt


-- loaded dataset from tf.keras.mnist.load_data()
-- Performed Normalization images reshaped((-1,2,28,1)) all images 28*28 pxl and 1 channel(grayScale)
---One hot encoded labels y_train,y_test using tf.keras.utils.to_categorical()

--Data_agumented and using it before i/p layer

--we have taken input_shape = (28,28,1) as mnist data set have grayscaled images that is channel is 1 and pxl = 28*28
--after every layer we need to do batch normalization as it stabilizes the layers output

--Trained model using the preprocessed training data.

--Utilized categorical cross-entropy as the loss function, stochastic gradient descent
   as the optimizer, and track accuracy as a performance metric.

--Evaluated the performance of the trained model using the testing
dataset, ensuring that the model generalizes well to unseen data.

