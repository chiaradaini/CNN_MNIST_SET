import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the VGG16 model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Save the model weights
model.save_weights('model_weights.h5')

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Create a new model with the same architecture
new_model = Sequential()
new_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Flatten())
new_model.add(Dense(128, activation='relu'))
new_model.add(Dense(10, activation='softmax'))

# Load the saved weights
new_model.load_weights('model_weights.h5')

# Compile the new model
new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the loaded model
loaded_score = new_model.evaluate(x_test, y_test, verbose=0)
print('Loaded model test loss:', loaded_score[0])
print('Loaded model test accuracy:', loaded_score[1])
