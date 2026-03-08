# Group 5: Elliot Mai, Cj Robinson, Adrienne Samos, Gerald Turner, Israel Vega
# Assignment: Unit 9 Assignment
# Date: Mar 9. 2026

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import time # to track training time

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Display the first 10 digit images from the dataset
plt.figure(figsize=(14, 5))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
    plt.title(f'Image {i + 1}')
plt.show()

# Create a Sequential model
# Steps: Conv2D, ReLU, MaxPooling2D, Flatten, Dense, Dropout, Dense
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # Repeat the convolutional and pooling layers as needed to increase the depth of the model

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
start_time = time.time()
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
training_time = time.time() - start_time

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=False)

# Baseline batch size results
print(f"\n--- Baseline Results with batch size = 32 ---\n")
print(f"Test accuracy: {test_acc:.4}")
print(f"Test loss: {test_loss:.4}")
print(f"Training time: {training_time:.2f} seconds")

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
predictions = model.predict(x_test)

# Display the first 10 digit images along with their corresponding predictions
plt.figure(figsize=(14, 5))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Image #{i + 1}\nDigit: {predictions[i].argmax()}')
    plt.axis('off')
plt.show()