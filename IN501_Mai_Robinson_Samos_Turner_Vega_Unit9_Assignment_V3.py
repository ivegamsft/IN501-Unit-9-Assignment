# Name: Israel Vega
# Assignment: Unit 9 Assignment
# Date: Mar 9. 2026

# Update some environment variables to suppress TensorFlow warnings and optimize performance
import os
import time
#suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Set the logging levels to error to reduce noise
tf.get_logger().setLevel('ERROR')

#Set the constants for the variables
# We can adjust these numbers to see how it affects the accuracy and loss of the model.
TRAINING_RUNS = 3 #5 # Number of epochs to train the model. Increase for better accuracy but will take longer to train.
BATCH_SIZE = 4 #32 # Number of samples per gradient update. Increase for better accuracy but will take longer to train. Decrease to speed up training but will reduce accuracy.
NUM_FILTERS = 32 # Increase to detect edges, decrease to speed up training but will reduce accuracy
POOL_SIZE = 2 # Size of the pooling window to look at. Decrease to provide more detail
NUM_STRIDES = 2 # Stride size for the pooling layer or pixel steps. Decrease to provide more detail but will increase training time.
LAYER_HIDDEN_DENSITY = 4 #64 # Number of neurons in the hidden layer. Increase for better accuracy but will take longer to train.
LAYER_OUTPUT_DENSITY = 10 # Set to 10 since the data set is 0-9 digits. DO NOT CHANGE THIS NUMBER.

# Load and preprocess the MNIST dataset
data_load_time = time.perf_counter()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
data_load_time_elapsed = time.perf_counter() - data_load_time

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#Show the Shape of the training data and the test data
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

# Display the first 10 digit images from the dataset. This is the baseline
plt.figure(figsize=(14, 5))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
    plt.title(f'Image {i+1}')
plt.show()

# Create a Sequential model
# Steps: Conv2D, ReLU, MaxPooling2D, Flatten, Dense, Dropout, Dense
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(filters=NUM_FILTERS, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=POOL_SIZE, strides=NUM_STRIDES),

    # Repeat the convolutional and pooling layers as needed to increase the depth of the model
    layers.Flatten(),
    layers.Dense(LAYER_HIDDEN_DENSITY, activation='relu'),
    layers.Dense(LAYER_OUTPUT_DENSITY, activation='softmax')
])

# Compile the model
model_compile_time = time.perf_counter()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_compile_time_elapsed = time.perf_counter() - model_compile_time
# Print the model summary to see the architecture and the number of parameters in each layer
print(model.summary())

# Train the model on the training data
model_train_time = time.perf_counter()
history = model.fit(x_train, y_train, epochs=TRAINING_RUNS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
model_train_time_elapsed = time.perf_counter() - model_train_time

# Evaluate the model on the test data
model_evaluate_time = time.perf_counter()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=False)
model_evaluate_time_elapsed = time.perf_counter() - model_evaluate_time

# Format the output as an image using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.legend()
plt.title('Accuracy', fontsize=16)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(plt.ylim())])
plt.legend()
plt.title('Loss', fontsize=16)
plt.show()

# Generate predictions based on the test data
model_predict_time = time.perf_counter()
predictions = model.predict(x_test)
model_predict_time_elapsed = time.perf_counter() - model_predict_time

# Display the first 10 digit images along with their corresponding predictions
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Image #{i + 1}\nDigit: {predictions[i].argmax()}\nConfidence: {predictions[i].max():.2f}', fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()

#Print the results so we can see how the model did
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')
print(f"Data loading time: {data_load_time_elapsed:.2f} seconds")
print(f"Model compilation time: {model_compile_time_elapsed:.2f} seconds")
print(f"Training time: {model_train_time_elapsed:.2f} seconds")
print(f"Evaluation time: {model_evaluate_time_elapsed:.2f} seconds")
print(f"Prediction time: {model_predict_time_elapsed:.2f} seconds")