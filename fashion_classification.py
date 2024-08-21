import keras # Import keras for building the neural network
from keras import datasets, layers, models # Import necessary modules from Keras, such as layers and models
import matplotlib.pyplot as plt # Import Matplotlib for visualizing the images

class FashionMNISTModel:
    def __init__(self):
        # Initialize the Fashion MNIST dataset
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_data()

    def load_data(self):
        # Load and preprocesses the Fashion MNIST dataset, which is split into training and testing sets
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        
        # Normalize the pixel values
        train_images, test_images = train_images / 255.0, test_images / 255.0
        
        # Reshape the data to include the channel dimension (28x28x1)
        # The '1' indicates a single color channel (grayscale)
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        
        return train_images, train_labels, test_images, test_labels

    def build_model(self):
        # Define the CNN model
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),            # Input layer specifying the input shape
            layers.Conv2D(32, (3, 3), activation='relu'),# First convolutional layer
            layers.MaxPooling2D((2, 2)),                # First max pooling layer
            layers.Conv2D(64, (3, 3), activation='relu'),# Second convolutional layer
            layers.MaxPooling2D((2, 2)),                # Second max pooling layer
            layers.Conv2D(64, (3, 3), activation='relu'),# Third convolutional layer
            layers.Flatten(),                           # Flatten layer to convert 2D to 1D
            layers.Dense(64, activation='relu'),        # Fully connected layer with 64 units
            layers.Dense(10, activation='softmax')      # Output layer with 10 units for classification
        ])
        
        # Compile the model with an optimizer, loss function, and accuracy metric
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Store the model in the class instance
        self.model = model

    def train_model(self, epochs=10): # An epoch is one complete pass through the entire training dataset
        # Train the model on the training data
        history = self.model.fit(self.train_images, self.train_labels, epochs=epochs, 
                                 validation_data=(self.test_images, self.test_labels))
        return history

    def evaluate_model(self):
        # Evaluate the model on the test  and print the accuracy
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels) 
        test_acc_percentage = test_acc * 100 # Convert accuracy to a percentage
        print(f'Test accuracy: {test_acc_percentage:.2f}%') # Print the test accuracy as a percentage
        return test_acc_percentage

    def predict(self, num_images=2):
        # Make predictions on the first few images in the test set
        predictions = self.model.predict(self.test_images[:num_images])
        
        # Print the predicted classes
        predicted_classes = predictions.argmax(axis=1)
        print('Predicted classes:', predicted_classes)
        
        # Print the actual classes for comparison
        actual_classes = self.test_labels[:num_images]
        print('Actual classes:', actual_classes)
        
        return predicted_classes, actual_classes

    def plot_samples(self, indices=[0, 1]):
        # Plot multiple images based on provided indices
        for i in indices:
            plt.figure(figsize=(3, 3))
            plt.imshow(self.test_images[i].reshape(28, 28), cmap='gray')
            plt.title(f'Actual: {self.test_labels[i]}, Predicted: {self.model.predict(self.test_images[i:i+1]).argmax(axis=1)[0]}')
            plt.axis('off')
            plt.show()

# Main function to execute the workflow
if __name__ == "__main__":
    # Instantiate the class
    fashion_mnist_model = FashionMNISTModel()

    # Defines and compiles the CNN model
    fashion_mnist_model.build_model()

    # Trains the CNN model on the training data
    fashion_mnist_model.train_model(epochs=10)

    # Evaluates the model on the test data and prints the accuracy
    fashion_mnist_model.evaluate_model()

    # Makes predictions on a specified number of test images
    fashion_mnist_model.predict(num_images=2)

    # Plot the predicted images with their actual and predicted labels
    fashion_mnist_model.plot_samples(indices=[0, 1])
