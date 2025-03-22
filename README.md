# Binary Image Classifier
This repository contains the code for a basic, binary image classifier.

## How to Run
Make sure all libraries in the `requirement.txt` file are installed. You can then run the program using `python3 image_classifier.py`. You will then be prompted to either train a model or load a model.

### Training Model Requirement
In order to train a model, you must have 2 folders, each title with the folders respective class label, containing images (for the folders class). ~40-50 images per should be enough for good but not perfect results. The program will purge any non-viable images (those that are corrupt or not of "jpeg", "jpg", "bmp", "png" extensions) from the folders. You will then follow the steps printed out in the terminal.

### Loading Model Requirements
To load a model, it must be a trained, tensorflow model with a name in the format `class1_class2_model.*extension*`. You will then follow the steps printed out in the terminal.