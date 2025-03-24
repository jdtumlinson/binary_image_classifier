"""
    File: image_classifier.py
    Developer: Joshua Tumlinson
    Data: March 21, 2025
    Primitives image classifer made using the python programming language and the tensorflow library

    Based on the tutorial by Nicholas Renotte https://www.youtube.com/watch?v=jztwpsIzEGc
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import data_cleaner
import numpy as np
import cv2
import os

def limit_memory_growth() -> None:
    """
        Limits memory growth in GPU's
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth set.")
        except RuntimeError as e:
            print(e)



def train_model():
    """Train a tensorflow model to classify two images

    Returns:
        model: trained tensorflow model
    """
    data_folder = input("Enter the relative path to directory containing the data (defaults to `data`): ")
    
    if data_folder == "": data_folder = "data"
    data_cleaner.cleanData(data_folder)
    
    class1, class2 = sorted([f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))])
    
    data = tf.keras.utils.image_dataset_from_directory(data_folder)

    viewStats = False
    viewStatsChoice = ""
    
    while viewStatsChoice not in ["y", "n"]:
        viewStatsChoice = input("View statistics? (Y or N): ").lower()
        
        if viewStatsChoice == "y":
            viewStats = True
            break
        elif viewStatsChoice == "n": break
        else: print("Input must be \"Y\" or \"N\"")

    data_iterator = data.as_numpy_iterator()

    batch = data_iterator.next()

    # Determine the which classifier is designated to which image group
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])

    plt.show()
    
    return pre_process_data(data, viewStats, class1, class2), class1, class2



def pre_process_data(data, viewStats: bool, class1: str, class2: str):
    """Partition datasets for training, validation and testing. Create model

    Args:
        data: image data
        viewStats (bool): choice to view statistics
    """
    #Normalize values
    data = data.map(lambda x, y: (x / 255, y))
    #batch = data.as_numpy_iterator().next()

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1) + 1

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    if viewStats:
        print("\n\n\n")
        print(f"Data size: {len(data)}, Training size: {len(train)}, Validation size: {len(val)}, Testing size: {len(test)}")


    #Create model
    model = tf.keras.models.Sequential()

    # Add layer | 
    #               16 filters, 3x3 size
    #               1 step size for filter movement
    #               ReLU activation
    #               Shape of 256 by 256 by 3 (256x256 image with 3 layers of colors -- red, green, blue)
    model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation="relu", input_shape=(256, 256, 3)))
    
    #Perform max pooling to downsize feature map
    model.add(tf.keras.layers.MaxPooling2D())

    #32 filters, 3x3 size...
    model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())

    #16 filters, 3x3 size..
    model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())

    #Convert 2D feature map into 1D vector
    model.add(tf.keras.layers.Flatten())

    #Fully connected layer with 256 neurons with ReLU activation
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    
    #Create an output layer with 1 neuron and sigmoid activation
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    #Optimizer: adam -- adapative moment estimation
    #Loss function: BinaryCrossentropy()
    #Use the accuracy model to evaluate performance
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    if viewStats: model.summary()
    
    return train_data(model, train, val, test, viewStats, class1, class2)



def train_data(model, train, val, test, viewStats: bool, class1: str, class2: str):
    """Function responsible for actually training the model and plotting statistics regarding model

    Args:
        model: set up tensorflow model
        train: training dataset
        val: validation dataset
        test: testing dataset
        viewStats (bool): choice to view statistics

    Returns:
        model: trained tensorflow model
    """
    logdir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

    if viewStats:
        #Plot losses
        fig = plt.figure()
        plt.plot(hist.history["loss"], color="blue", label="loss")
        plt.plot(hist.history["val_loss"], color="red", label="val_loss")
        fig.suptitle("Losses", fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        #Plot accuracy
        fig = plt.figure()
        plt.plot(hist.history["accuracy"], color="blue", label="loss")
        plt.plot(hist.history["val_accuracy"], color="red", label="val_loss")
        fig.suptitle("accuracy", fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        #Precision, recall and binary accuracy
        pre = tf.keras.metrics.Precision()
        re = tf.keras.metrics.Recall()
        acc = tf.keras.metrics.BinaryAccuracy()

        for batch in test.as_numpy_iterator():
            x, y = batch
            yhat = model.predict(x)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)

        print(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, BinaryAccuracy: {acc.result().numpy()}")
        
    model.save(os.path.join("models", f"{class1}_{class2}_model.keras"))
    
    return model



def load_model():
    """Loads a model from the `models` folder to be used for classification


    Raises:
        LookupError: model must exist

    Returns:
        model: loaded-trained tensorflow model
    """
    modelChoice = input("Enter a model name from the `models` folder: ")
    class1, class2 = modelChoice.split("_")[:-1]
    
    modelPath = os.path.join("models", modelChoice)
    if os.path.exists(modelPath) == False: raise LookupError("Model doesn't exist")
    
    model = tf.keras.models.load_model(modelPath)

    return model, class1, class2



def predict(mcc: tuple):
    """Classifies a given image based on a trained model

    Args:
        mcc (tuple): model (m) trained tensorflow model, classes 1 and 2(c)
    """
    model, class1, class2 = mcc
    
    while True:
        imageChoice = input("\n\nEnter the relative file path for the image you would like to classify: ")
    
        img = cv2.imread(imageChoice)
        resize = tf.image.resize(img, (256, 256))

        yhat = model.predict(np.expand_dims(resize / 255, 0))

        if yhat < 0.5: print(f"\nThis image is a {class2}")
        else: print(f"\nThis image is a {class1}")
        
        choice = ""
        
        while choice not in ["y", "n"]:
            choice = input("\n\nWould you like to continue predicting images? (Y or N): ").lower()
            
            if choice == "y": continue
            elif choice == "n": return
            else: print("Input must be \"Y\" or \"N\"")



def __main__():
    limit_memory_growth()

    userDecision = ""
    
    while userDecision not in ["t", "l"]:
        userDecision = input("Train (T) or load (L) data?: ").lower()
        
        if userDecision == "t": model = predict(train_model())
        elif userDecision == "l": model = predict(load_model())
        else: print("Input must be \"T\" or \"L\"")
 
        

__main__()
