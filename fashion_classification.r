# Load necessary libraries
library(keras)
library(R6)

# Define the FashionMNISTModel class
FashionMNISTModel <- R6Class(
  "FashionMNISTModel",
  
  public = list(
    # Constructor method to initialize the dataset
    initialize = function() {
      data <- self$load_data()
      self$train_images <- data$train_images
      self$train_labels <- data$train_labels
      self$test_images <- data$test_images
      self$test_labels <- data$test_labels
    },
    
    # Method to load and preprocess the data
    load_data = function() {
      # Load the Fashion MNIST dataset
      fashion_mnist <- dataset_fashion_mnist()
      train_images <- fashion_mnist$train$x
      train_labels <- fashion_mnist$train$y
      test_images <- fashion_mnist$test$x
      test_labels <- fashion_mnist$test$y
      
      # Normalize the pixel values
      train_images <- train_images / 255
      test_images <- test_images / 255
      
      # Reshape the data to include the channel dimension
      train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
      test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))
      
      list(train_images = train_images, train_labels = train_labels,
           test_images = test_images, test_labels = test_labels)
    },
    
    # Method to build the CNN model
    build_model = function() {
      # Define the CNN model
      model <- keras_model_sequential() %>%
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
        layer_flatten() %>%
        layer_dense(units = 64, activation = 'relu') %>%
        layer_dense(units = 10, activation = 'softmax')
      
      # Compile the model
      model %>% compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = c('accuracy')
      )
      
      # Store the model in the class instance
      self$model <- model
    },
    
    # Method to train the model
    train_model = function(epochs = 10) {
      # Train the model on the training data
      history <- self$model %>% fit(
        self$train_images, self$train_labels,
        epochs = epochs, validation_split = 0.2
      )
      return(history)
    },
    
    # Method to evaluate the model
    evaluate_model = function() {
      # Evaluate the model on the test data
      results <- self$model %>% evaluate(self$test_images, self$test_labels)
      
      # Convert accuracy to a percentage
      test_acc_percentage <- results[2] * 100
      
      # Print the test accuracy as a percentage
      cat(sprintf("Test accuracy: %.2f%%\n", test_acc_percentage))
      
      return(test_acc_percentage)
    },
    
    # Method to make predictions on test data
    predict = function(num_images = 2) {
      # Make predictions on the specified number of test images
      predictions <- self$model %>% predict(self$test_images[1:num_images,,,drop = FALSE])
      
      # Get the predicted classes
      predicted_classes <- apply(predictions, 1, which.max) - 1
      
      # Print the predicted and actual classes
      cat("Predicted classes:", predicted_classes, "\n")
      cat("Actual classes:", self$test_labels[1:num_images], "\n")
      
      return(list(predicted = predicted_classes, actual = self$test_labels[1:num_images]))
    }
  ),
  
  private = list(
    # Private variables to store the data and model
    train_images = NULL,
    train_labels = NULL,
    test_images = NULL,
    test_labels = NULL,
    model = NULL
  )
)

# Main function to execute the workflow
main <- function() {
  # Instantiate the class
  fashion_mnist_model <- FashionMNISTModel$new()
  
  # Build the CNN model
  fashion_mnist_model$build_model()
  
  # Train the model
  fashion_mnist_model$train_model(epochs = 10)
  
  # Evaluate the model
  fashion_mnist_model$evaluate_model()
  
  # Make predictions on a few images
  fashion_mnist_model$predict(num_images = 2)
}

# Run the main function
main()
