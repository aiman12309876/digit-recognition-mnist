import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                    validation_data=(x_test, y_test), verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("=" * 60)
print("HANDWRITTEN DIGIT RECOGNITION")
print("=" * 60)
print()
print("Dataset: MNIST")
print("Training Samples:", len(x_train))
print("Testing Samples:", len(x_test))
print("Classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
print()
print("Model Architecture:")
print("  Conv2D(32) + MaxPooling")
print("  Conv2D(64) + MaxPooling")
print("  Conv2D(64)")
print("  Flatten + Dense(64) + Dense(10)")
print()
print("Test Accuracy:", round(test_acc * 100, 2), "%")
print()
print("=" * 60)
print("MODEL READY")
print("=" * 60)

for i in range(5):
    sample = x_test[i].reshape(28, 28)
    pred = model.predict(x_test[i].reshape(1, 28, 28, 1), verbose=0)
    predicted_digit = np.argmax(pred)
    actual_digit = np.argmax(y_test[i])
    print(f"Sample {i+1}: Predicted: {predicted_digit}, Actual: {actual_digit}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()