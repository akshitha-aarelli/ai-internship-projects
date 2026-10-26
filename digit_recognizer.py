"""Project 1: Handwritten Digit Recognizer using a CNN on the MNIST dataset."""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data (downloads automatically the first time)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess: scale to 0-1 and add channel dimension
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

# 3. Build the CNN
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# 4. Train
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# 5. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")

# 6. Show a few predictions
preds = model.predict(x_test[:10])
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title(f"Pred: {np.argmax(preds[i])}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("predictions.png")
plt.show()

model.save("digit_cnn.keras")