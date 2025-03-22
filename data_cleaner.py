"""
    File: data_cleaner.py
    Developer: Joshua Tumlinson
    Data: March 21, 2025
    Remove any non-viable images from a data directory
"""

import imghdr
import cv2
import os



def cleanData(data_dir = "data") -> None:
    """Remove non-viable data from a given directory

    Args:
        data_dir (str, optional): relative location of directory. Defaults to "data".

    Raises:
        ValueError: file must have 2 folders within for class seperation
    """
    #Retrieve files from data folder
    data_dir_files = os.listdir(data_dir)

    #Remove MacOS specific folder
    if ".DS_Store" in data_dir_files: data_dir_files.remove(".DS_Store")

    #Verify that theres only two folders of classes
    if len(data_dir_files) != 2:
        raise ValueError("Image classifier only works with two (2) classes")

    #Valid image extensions
    image_exts = ["jpeg", "jpg", "bmp", "png"]

    print("Cleaning non-viable images...")

    #Clean up any images that won't work for this program
    for image_class in data_dir_files:
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f"Image not in ext list {image_path}")
                    os.remove(image_path)
                else:
                    size_kb = os.path.getsize(image_path) / 1024
                    if size_kb <= 10: 
                        print(f"Image size is to small {image_path}")
                        os.remove(image_path)
                
            except Exception as e:
                print(f"Issue with image {image_path}")
        
        print(f"Number of files in {data_dir}: {len(os.listdir(os.path.join(data_dir, image_class)))}")
    
    return