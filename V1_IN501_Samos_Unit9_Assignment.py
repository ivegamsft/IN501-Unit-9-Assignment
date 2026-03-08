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

# Adrienne's modification: Increase batch size
# Baseline used default batch size (32) - Adrienne is changing it to 64
BATCH_SIZE = 64

# Create the same Sequential model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with Adrienne's batch size
start_time = time.time()
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test))
training_time = time.time() - start_time

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=False)

# Modified batch size results for comparison
print(f"\n--- Results with batch size = {BATCH_SIZE} ---\n")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Training time: {training_time:.2f} seconds")

# Generate predictions for first 10 test images
predictions = model.predict(x_test[:10])

# Display the first 10 test images with predictions
plt.figure(figsize=(14, 5))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
    plt.title(f'Pred: {predictions[i].argmax()}\nTrue: {y_test[i]}')
plt.suptitle(f'Predictions with Batch Size = {BATCH_SIZE}')
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()