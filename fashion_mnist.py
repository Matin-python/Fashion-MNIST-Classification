import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Flatten

import matplotlib.pyplot as plt


def imgshow(img, title=None):
    plt.close()
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


# Load Fashion-MNIST dataset

fashion = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion.load_data()


# Class names

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


# Display dataset information

print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

print("First label:", train_labels[0])
imgshow(
    train_images[0],
    class_names[train_labels[0]]
)


# Normalize images

train_images = train_images / 255.0
test_images = test_images / 255.0


# Build neural network

model = Sequential([
    tf.keras.Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10)
])


# Compile model

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    optimizer='adam',
    metrics=['accuracy']
)


# Display model architecture

model.summary()


# Train model

h = model.fit(
    train_images,
    train_labels,
    epochs=20,
    validation_split=0.1
)


# Evaluate model

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# Make predictions

y_pred = model.predict(test_images)

predicted_labels = tf.argmax(y_pred, axis=1)


# Display prediction

print("Predicted:", class_names[predicted_labels[0].numpy()])
print("Actual:", class_names[test_labels[0]])

imgshow(
    test_images[0],
    f"Predicted: {class_names[predicted_labels[0].numpy()]}\n"
    f"Actual: {class_names[test_labels[0]]}"
)


# Plot training loss

plt.close()
plt.plot(h.history['loss'], label='Training Loss')
plt.plot(h.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot training accuracy

plt.close()
plt.plot(h.history['accuracy'], label='Training Accuracy')
plt.plot(h.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
