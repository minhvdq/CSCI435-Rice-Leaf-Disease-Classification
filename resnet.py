import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import argparse

import torch
import torchvision.models as models
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Precision, Recall

import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchmetrics

import cv2 as cv

import skopt
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize


# --- Global Constants (Kept the same) ---
num_classes = 5
batch_size = 60 # Note: This is overridden by the BO process

labels = ['Brown Spot', 'Leaf Scaled', 'Rice Blast', 'Rice Turgor', 'Sheath Blight']

data_root = "./Dhan-Shomadhan"

IMG_WIDTH = 256
IMG_HEIGHT = 256

MODEL_SAVE_PATH = 'cnn_model_state.pt'
OPTIMIZED_HPS_PATH = 'optimized_hps.pkl'

# You could calculate your dataset's specific mean/std for better results.
MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToTensor(),             # Converts image to tensor and scales to [0, 1]
    transforms.RandomHorizontalFlip(p=0.5), # Augmentation 1: Random flip
    transforms.RandomRotation(degrees=15),  # Augmentation 2: Small random rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Augmentation 3: Minor color variation
    transforms.Normalize(MEAN, STD)    
])

test_val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)   
])

class CustomImageDataset(Dataset):
    def __init__(self, file_paths, targets, transform=None):
        self.file_paths = file_paths
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        # 1. Load image (I/O operation happens here)
        img_path = self.file_paths[index]
        image = cv.imread(img_path)
        
        # Check if image was loaded correctly
        if image is None:
            print(f"Warning: Could not load image {img_path}. Returning zero tensor.")
            return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32), self.targets[index]

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # 2. Apply preprocessing (Rotation and Resizing from your original logic)
        img_h, img_w, _ = image.shape
        
        # Rotation logic
        if img_h > img_w:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

        if image.shape[0] != IMG_HEIGHT or image.shape[1] != IMG_WIDTH:
            image = cv.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)

        # 3. Apply standard PyTorch/Tensor transformations
        if self.transform:
            image = self.transform(image)
        
        target = self.targets[index]
        return image, target

# --- EarlyStopper (Kept the same) ---
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  
            return False
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

# --- Helper Functions (Kept the same) ---
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_all_paths(data_root):
    all_paths = []
    all_labels = []

    condition_dirs = [d.name for d in os.scandir(data_root) if d.is_dir()]
    condition_dirs.sort()
    for condition in condition_dirs:
        condition_path = os.path.join(data_root, condition)
        print(f"Condition path is {condition_path}")
        label_paths = [f.name for f in os.scandir(condition_path) if f.is_dir()]
        label_paths.sort()
        for index, label_path in enumerate(label_paths):
            if(label_path.startswith('.')):
                continue
            label_path_full = os.path.join(condition_path, label_path)
            image_file_paths = [f.name for f in os.scandir(label_path_full) if f.is_file()]
            image_file_paths.sort()
            for image_file_path in image_file_paths:
                image_file_path_full = os.path.join(label_path_full, image_file_path)
                all_paths.append(image_file_path_full)
                all_labels.append(index)
    return all_paths, all_labels


def split_data(root_path, train_ratio=0.75, val_ratio=0.15):
    
    all_paths, all_labels = get_all_paths(root_path)
    print(f"get {len(all_paths)} paths and {len(all_labels)} labels")

    indices = np.arange(len(all_paths))
    np.random.shuffle(indices)
    data_tr, data_val, data_t, labels_tr, labels_val, labels_t = [], [], [], [], [], []
    for ci, index in enumerate(indices):
        if ci < int(len(indices) * train_ratio):
            data_tr.append(all_paths[index])
            labels_tr.append(all_labels[index])
        elif ci < int(len(indices) * (train_ratio + val_ratio)):
            data_val.append(all_paths[index])
            labels_val.append(all_labels[index])
        else:
            data_t.append(all_paths[index])
            labels_t.append(all_labels[index])
    return data_tr, data_val, data_t, labels_tr, labels_val, labels_t


def train():
    model = model.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    batch_size = 32
    num_epoches = 20

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_tr, data_val, data_t, labels_tr, labels_val, labels_t = split_data(data_root)

    train_dataset = CustomImageDataset(data_tr, labels_tr, transform=transform)
    val_dataset = CustomImageDataset(data_val, labels_val, transform=transform)
    test_dataset = CustomImageDataset(data_t, labels_t, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)



