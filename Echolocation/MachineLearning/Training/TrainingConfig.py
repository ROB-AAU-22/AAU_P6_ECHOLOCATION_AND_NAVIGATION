#!/usr/bin/env python3

# Hyperparameters and configuration settings
DISTANCE_THRESHOLD_ENABLED = True
DISTANCE_THRESHOLD = 2
DISTANCE_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.95]

# Configuration for regressor
# Configuration for regressor
REGRESSOR_LEARNING_RATES = [0.01]  # List of learning rates to try for the regressor
REGRESSOR_HIDDEN_DIMS = [[256,256]]  # List of hidden layer dimensions for the regressor (each list is a layer configuration)
REGRESSOR_BATCH_SIZES = [64]  # List of batch sizes to try for the regressor
REGRESSOR_NUM_LAYERS_LIST = [2]  # List of possible numbers of layers for the regressor
REGRESSOR_WEIGHT_DECAYS = [1e-5]  # List of weight decay values for regularization
REGRESSOR_LAYER_TYPE = ["ReLU"]  # List of activation functions to use in the regressor (e.g., "ReLU", "Tanh")
REGRESSOR_OPTIMIZER = ["Adam"]  # List of optimizers to try for the regressor

# Configuration for classifier
# Configuration for classifier
CLASSIFIER_LEARNING_RATES = [0.01]  # List of learning rates to try for the classifier
CLASSIFIER_HIDDEN_DIMS = [512]  # List of hidden layer dimensions for the classifier
CLASSIFIER_BATCH_SIZES = [64]  # List of batch sizes to try for the classifier
CLASSIFIER_NUM_LAYERS_LIST = [2]  # List of possible numbers of layers for the classifier
CLASSIFIER_WEIGHT_DECAYS = [1e-5]  # List of weight decay values for regularization
CLASSIFIER_LAYER_TYPE = ["ReLU"]  # List of activation functions to use in the classifier (e.g., "ReLU", "Tanh")
CLASSIFIER_OPTIMIZER = ["Adam"]  # List of optimizers to try for the classifier


CLASSIFICATION_THRESHOLD = 0.4
PATIENCE = 10
PLOT_DPI = 200
NUM_EPOCHS = 200
