# Fashion MNIST Classification
*BAN6420 Module 5 Assignment*

## Overview
The Project demonstrates how to implement a Convolutional Neural Network (CNN) using both Python and R to classify images from the Fashion MNIST dataset into the correct categories. The goal is to use the CNN to identify various fashion items from the dataset and make predictions. The code is provided in both Python and R to demonstrate the implementation in two different environments.

## Dataset
The dataset used in this project is the Fashion MNIST dataset which is a collection of 70,000 grayscale images of 10 different classes of clothing items. Each image is 28x28 pixels in size.

## Understanding the Code
#### Data Loading and Preprocessing 
The Fashion MNIST dataset is loaded and preprocessed by normalizing pixel values and reshaping the images.
#### Model Building 
A CNN with three convolutional layers followed by two dense layers is built using Keras.
#### Training 
The model is trained on the training data for 10 epochs with validation on the test data.
#### Evaluation and Prediction 
The model's performance is evaluated on the test set, and predictions are made for two test images.

## Interpretation of the Visualization
#### Image Representation 
Each image in the Fashion MNIST dataset represents a grayscale image of a fashion item, such as a shoe, shirt, or bag. The images are 28x28 pixels in size, and the pixel values range from 0 to 255, with 0 being black and 255 being white.

#### Predicted vs. Actual Labels
The title of each plot shows both the predicted and the actual class labels. The actual class is the correct label from the dataset, while the predicted class is the label assigned by the trained CNN model.
For example, if an image of a shoe is displayed and the title reads [Actual: 7, Predicted: 7] this indicates that the model correctly predicted the image as a "Shoe" (class 7). </br>
However, if the title reads [Actual: 5, Predicted: 2] this indicates that the model misclassified the image, predicting it as a "Pullover" (class 2) instead of a "Sandal" (class 5).

## Running the Script
### Python
- The code can be executed by running the Python script.
- The *FashionMNISTModel* class handles loading the dataset, building the CNN, training the model, evaluating the model, and making predictions.
- The script will train the model for 10 epochs (an epoch is one complete pass through the entire training dataset), evaluate it on the test dataset, and make predictions on the test images.
- The test accuracy is printed as a percentage after evaluation.

### R
- To run the R script, use the *source("fashion_classification.r")* command in your R environment.
- The *FashionMNISTModel* class in R handles similar tasks as in Python for loading the dataset, building the CNN, training the model, evaluating the model, and making predictions.
- The script will train the model for 10 epochs (an epoch is one complete pass through the entire training dataset), evaluate it on the test dataset, and make predictions on the first two test images.
- The test accuracy is displayed as a percentage after evaluation.

## Conclusion
This project demonstrates how to implement a CNN for image classification using the Fashion MNIST dataset in both Python and R. By following the instructions and understanding the code, you should be able to train a CNN model, evaluate its performance, and make predictions on unseen data. This project demonstrates the power of neural networks in image classification and provides a foundation for adapting the code for classifying user profile images for targeted marketing.
