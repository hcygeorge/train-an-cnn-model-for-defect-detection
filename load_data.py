# This module Define how to load and process datasets
#%%
import os
import pickle
import random
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torchvision import datasets, transforms

#%% Define how to get a list of image paths and a list of image labels
class CreateList():
    """Define how to get a list of image paths and a list of image labels.
    
    Args:
        dir_img (str): The directory that stores images.
        path_label (str): The file(usually a txt or csv) that records image labels.
        header (bool): Whether to skip the first line of file(path_label).
        shuffle (bool): Whether to shuffle the list of line in the file(path_label).
        train (bool): True if the dataset belongs to training data,
            and create_list() won't expect image labels.
        
    Attributes:
        dir_img (str): The directory that stores images.
        path_label (str): The file(usually a txt or csv) that records image labels.
    """
    def __init__(self, dir_img, path_label=None, header=True, shuffle=False, train=True):
        self.dir_img = dir_img
        self.path_label = path_label
        self.create_list(header, shuffle, train)
        
    def enocode_label(self):
        """Create dict to encode and decode labels to integer."""
        # Load labels from file(csv, txt,...)
        with open(self.path_label, 'r') as f:
            label_names = f.readlines()
            
        # Create label encoder and decoder
        self.label2code = {}
        self.code2label = {}
        
        for idx, label in enumerate(label_names):
            self.label2code[label[:-1]] = idx
            self.code2label[idx] = label[:-1]
            
        # Encoder image labels as integer
        label_new = []
        for label in self.label:
            label_new.append(self.label2code[label])
        
    def create_list(self, header, shuffle, train):
        """Collect image paths and of dataset.
        
        Parameters:
            header (bool): Whether to skip the first line of file(path_label).
            shuffle (bool): Whether to shuffle the list of line in the file(path_label).
            train (bool): True if the dataset belongs to training data,
                and create_list() won't expect image labels.
            
        Returns:
            img (list): List of image paths of the dataset.
            label (list): List of label code of the dataset.
        
        """
        with open(self.path_label, 'r') as f:
            if header:
                f.readline()  # skip header(first line)
            lines = f.readlines()

        if shuffle:
            random.shuffle(lines)
            
        self.img = []
        self.label = []
        self.filename = []
        
        for line in lines:
            line = line.split(',')
            self.filename.append(line[0])
            self.img.append(self.dir_img + line[0])
            if train:
                self.label.append(int(line[-1][0]))

        # Check whether image paths are valid.
        for path in self.img:
            if not os.path.exists(path):
                print("{} doesn't exist.".format(path.split('/')[-1]))
        
        # Count number of imgs
        self.length = len(self.img)
                
#%% Define a custom torch.Dataset
class CustomDataset(torch.utils.data.Dataset):
    """Define a custom torch.Dataset."""
    def __init__(self, img_list, label_list=None,
                 transform=None):
        """Load the dataset.
        
        Args:
            img_list (str): List of images
            label_list (list): List of labels
            transform (callable): A function/transform that takes in a
                PIL.Image and transforms it.
        """
        self.data = img_list
        self.label = label_list
        self.transform = transform

    def __getitem__(self, index):
        """Define how to load single sample(image and label) from list of paths.
        Args:
            index (int): Index.
            
        Returns:
            image (PIL.Image): Image of the given index.
            target (str): target of the given index.
        """
        # load one sample in a time
        image = self.data[index]
        if self.label != None:
            target = self.label[index]
        else:
            target = None
        image = Image.open(image)
        # turn image to RGB format if it is a grayscale
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        # preprocessing data
        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        """Compute number of data."""
        return len(self.data)
        

#%% Define image preprocessing and create a torch.DataLoader based on a torch.Dataset.
if __name__ == '__main__':
    # Paths
    dir_img_train = 'C:/Dataset/AOI/train_images/'
    path_label_train = 'C:/Dataset/AOI/train.csv'

    # Split image list and label list into train and valid.
    train_list = CreateList(dir_img_train, path_label_train, shuffle=True)
    train_valid_split = round(train_list.length * 0.8)
    train_img = train_list.img[:train_valid_split]
    train_label = train_list.label[:train_valid_split]
    valid_img = train_list.img[train_valid_split:]
    valid_label = train_list.label[train_valid_split:]

    # Image preprocessing
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    }

    # Create DataLoader
    train_dataset = CustomDataset(train_img,
                                  train_label,
                                  transform['train'])
    valid_dataset = CustomDataset(valid_img,
                                  valid_label,
                                  transform['valid'])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=48,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                            batch_size=48,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True)

    