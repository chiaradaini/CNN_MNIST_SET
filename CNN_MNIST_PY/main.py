import numpy as np
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt

  # Load training data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[:2500]
Y_train = Y_train[:2500]
X_test = X_test[:500]
Y_test = Y_test[:500]

# # Preprocess the data
# X_train = X_train.reshape(-1, 28, 28, 1)# Normal reshape for Neural Network feeding
# X_test = X_test.reshape(-1, 28, 28, 1)
# Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
# Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=10)

#Plot the first 9 images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.axis('off')

# Show the plot
plt.show()

# Define the network
layers = [
  ConvolutionLayer(16,3), # layer with 8 3x3 filters, output (26,26,16)
  MaxPoolingLayer(2), # pooling layer 2x2, output (13,13,16)
  SoftmaxLayer(13*13*16, 10) # softmax layer with 13*13*16 input and 10 output
  ] 

for epoch in range(1):
  print('Epoch {} ->'.format(epoch+1))
  # Shuffle training data
  permutation = np.random.permutation(len(X_train))
  X_train = X_train[permutation]
  Y_train = Y_train[permutation]
  # Training the CNN
  loss = 0
  accuracy = 0
  for i, (image, label) in enumerate(zip(X_train, Y_train)):
    if i % 100 == 0: # Every 100 examples
      print("Step {}. For the last 100 steps: average loss {}, accuracy {}%".format(i+1, loss/100, accuracy))
      loss = 0
      accuracy = 0
    loss_1, accuracy_1 = CNN_training(image, label, layers)
    loss += loss_1
    accuracy += accuracy_1
  
# Testing the CNN
loss_test = 0
accuracy_test = 0
for im, label in zip(X_test, Y_test):
   _, loss_1, accuracy_1 = CNN_forward(im, label, layers)
   loss_test += loss_1
   accuracy_test += accuracy_1

num_test = len(X_test)
print('Test loss:', loss_test / num_test)
print('Test accuracy:', accuracy_test / num_test)
