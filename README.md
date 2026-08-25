# 👕 Fashion-MNIST Classification (Deep Learning)

Classifying clothing images using a **Neural Network** and the **Fashion-MNIST dataset** with **TensorFlow and Keras**. This project demonstrates how a simple deep learning model can classify grayscale images into 10 different clothing categories.

## Overview

This project uses a **Fully Connected Neural Network** to classify images from the Fashion-MNIST dataset.

The dataset contains grayscale images of clothing items, with each image having a resolution of **28×28 pixels**.

The model consists of:

* A `Flatten` layer to convert each 28×28 image into a one-dimensional vector
* A `Dense` layer with 128 neurons and ReLU activation
* A `Dense` output layer with 10 neurons, representing the 10 clothing classes

The model is trained for **20 epochs** using the **Adam optimizer** and **Sparse Categorical Crossentropy** loss function.

## Features

* 🧠 Neural Network classification
* 👕 Fashion-MNIST dataset
* 🖼️ 28×28 grayscale image processing
* 🔄 Image flattening using a `Flatten` layer
* 🔥 ReLU activation function
* 🎯 10-class classification
* ⚙️ Adam optimizer
* 📉 Sparse Categorical Crossentropy loss
* 📊 Accuracy evaluation
* 📈 Training over multiple epochs
* 🧩 Neural network architecture visualization
* 🖼️ Displaying dataset images

## Technologies Used

* Python 3
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Keras Visualizer

## Dataset

The project uses the **Fashion-MNIST dataset**, which is available directly through TensorFlow/Keras.

The dataset contains:

* **60,000 training images**
* **10,000 test images**
* **28×28 grayscale images**
* **10 clothing categories**

The classes are:

| Label | Class      |
| ----- | ---------- |
|     0 | T-shirt    |
|     1 | Trouser    |
|     2 | Pullover   |
|     3 | Dress      |
|     4 | Coat       |
|     5 | Sandal     |
|     6 | Shirt      |
|     7 | Sneaker    |
|     8 | Bag        |
|     9 | Ankle boot |

Each image contains pixel values ranging from **0 to 255**, which are normalized before training.

## Deep Learning Workflow

1. Load the Fashion-MNIST dataset.
2. Split the dataset into training and testing sets.
3. Display sample images from the dataset.
4. Normalize the image pixel values.
5. Build a neural network using Keras.
6. Flatten each 28×28 image.
7. Train the model for 20 epochs.
8. Predict the classes of test images.
9. Compare the predictions with the actual labels.
10. Visualize the neural network architecture.

## Model Architecture

The neural network consists of:

```text
Input Image
   │
   ▼
28 × 28 Image
   │
   ▼
Flatten
   │
   ▼
128 Neurons
ReLU Activation
   │
   ▼
10 Output Neurons
   │
   ▼
Class Prediction
```

The model is implemented using Keras:

```python
model = Sequential()

model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10))
```

## Model Compilation

The model uses:

```python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    optimizer='adam',
    metrics=['accuracy']
)
```

The model is trained for:

```python
h = model.fit(
    train_images,
    train_labels,
    epochs=20
)
```

## Project Structure

```text
Fashion-MNIST-Classification/
│
├── screenshots/
│   ├── prediction.png
│   ├── train_image.png
│   ├── Training_and_Validation_Accuracy.png
│   └── Training_and_Validation_Loss.png
│
├── fashion_mnist.py
├── fashion_mnist.ipynb
├── DL_model.png
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Matin-python/Fashion-MNIST-Classification.git
```

Move into the project directory:

```bash
cd Fashion-MNIST-Classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

or install them manually:

```bash
pip install tensorflow keras matplotlib keras-visualizer
```

## How to Run

Run the Python script:

```bash
python fashion_mnist.py
```

The program will:

* Download the Fashion-MNIST dataset if it is not already available.
* Display sample images.
* Build the neural network.
* Display the model architecture.
* Train the model for 20 epochs.
* Predict test images.

## Evaluation

The model uses **accuracy** as its main evaluation metric.

During training, Keras displays the training loss and accuracy for each epoch.

The model can then be used to predict the clothing category of test images:

```python
y_pred = model.predict(test_images)
```

The predicted output contains scores for each of the 10 classes.

## Example Prediction

For example, the model can receive an image such as:

```text
28 × 28 grayscale image
        ↓
    Neural Network
        ↓
   10 class scores
        ↓
Predicted clothing class
```

The actual label can be checked using:

```python
print(test_labels[0])
```

and the corresponding image can be displayed using:

```python
imgshow(test_images[0])
```


## Future Improvements

* 📊 Add test-set accuracy evaluation
* 📉 Plot training and validation loss
* 📈 Plot training and validation accuracy
* 🎯 Add a confusion matrix
* 📋 Add a classification report
* 🖼️ Display predicted images with their class names
* 🧠 Experiment with different numbers of neurons
* 🔄 Add validation data
* 🧩 Experiment with different neural network architectures
* 🖥️ Build a graphical user interface
* 💾 Save and load the trained model
* 🧠 Replace the fully connected network with a CNN
* 📷 Add custom image prediction

## Contributing

Contributions, suggestions, and bug reports are welcome. Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
