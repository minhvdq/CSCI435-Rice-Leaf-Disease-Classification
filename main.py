import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import argparse

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Precision, Recall


# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics
import torchmetrics

import cv2 as cv

import skopt
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

num_classes = 5
batch_size = 60

labels = ['Brown Spot', 'Leaf Scaled', 'Rice Blast', 'Rice Turgor', 'Sheath Blight']

data_root = "./Dhan-Shomadhan"

# IMG_WIDTH = 200
# IMG_HEIGHT = 100

IMG_WIDTH = 256
IMG_HEIGHT = 256

MODEL_SAVE_PATH = 'cnn_model_state.pt'
OPTIMIZED_HPS_PATH = 'optimized_hps.pkl'


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
            # Handle error (e.g., return a black image or skip, depending on your error handling policy)
            print(f"Warning: Could not load image {img_path}. Returning zero tensor.")
            return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32), self.targets[index]

        # 2. Apply preprocessing (Rotation and Resizing from your original logic)
        img_h, img_w, _ = image.shape
        
        # Rotation logic
        if img_h > img_w:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

        # Resizing (using INTER_AREA for downsampling optimization)
        if image.shape[0] != IMG_HEIGHT or image.shape[1] != IMG_WIDTH:
            image = cv.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)

        # 3. Apply standard PyTorch/Tensor transformations
        if self.transform:
            image = self.transform(image)
        
        # PyTorch generally expects C x H x W format and float type (0.0 - 1.0)
        # Assuming transform (like torchvision.transforms.ToTensor) handles this.

        target = self.targets[index]
        return image, target

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None  # Corrected typo and initialized to None
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


'''
    activation functions can be reLu, Sigmoid, or Tanh.
'''

class CNN(nn.Module):
    def __init__(self, neurons, in_channels, activation_fn_str, layers1, layers2, kernel_size_1, kernel_size_2, dropout_rate, normalization, num_classes, img_h, img_w):
        super(CNN, self).__init__()
        
        # Helper to select activation function
        if activation_fn_str == 'relu': self.activation = nn.ReLU()
        elif activation_fn_str == 'sigmoid': self.activation = nn.Sigmoid()
        elif activation_fn_str == 'tanh': self.activation = nn.Tanh()
        else: self.activation = nn.ReLU()

        # Initial Block: Input Channels -> Neurons
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, neurons, kernel_size=kernel_size_1, padding=1),
            nn.BatchNorm2d(neurons) if normalization == 1 else nn.Identity(),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 1 (Repeated Conv-Pool)
        layers_1_list = []
        for _ in range(layers1):
            layers_1_list.append(nn.Conv2d(neurons, neurons, kernel_size=kernel_size_1, padding='same')) # Use 'same' for simplicity
            layers_1_list.append(self.activation)
            layers_1_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block1 = nn.Sequential(*layers_1_list)
        
        # Block 2 (Repeated Conv-Pool) - Use different kernel size if desired
        layers_2_list = []
        if dropout_rate > 0:
            layers_2_list.append(nn.Dropout2d(p=dropout_rate))

        cur_channel = neurons
        for _ in range(layers2):
            layers_2_list.append(nn.Conv2d(cur_channel, cur_channel * 2, kernel_size=kernel_size_2, padding='same')) # Double filters for deeper layers
            layers_2_list.append(self.activation)
            layers_2_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            cur_channel = cur_channel * 2

        self.block2 = nn.Sequential(*layers_2_list)
        
        self.fc1 = nn.Linear(cur_channel, num_classes) # Simplified to direct classification

    def forward(self, x):
        x = self.initial_block(x)
        x = self.block1(x)
        x = self.block2(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def train_and_evaluate(hyperparameters, train_loader, val_loader, device):
    neurons, activation_str, layers1, layers2, kernel_size, dropout_rate, normalization, lr, batch_size, num_epochs = hyperparameters

    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    layers1 = int(layers1)
    layers2 = int(layers2)
    neurons = int(neurons)
    kernel_size = int(kernel_size)

    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    best_val_accuracy = -np.inf

    model = CNN(neurons=neurons, in_channels=3, activation_fn_str=activation_str, layers1=layers1, layers2=layers2, kernel_size_1=kernel_size, kernel_size_2=kernel_size, dropout_rate=dropout_rate, normalization=normalization, num_classes=num_classes, img_h=IMG_HEIGHT, img_w=IMG_WIDTH).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        val_loss_sum = 0
        val_sample_count = 0
        val_acc_metric.reset()
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_sample_count += images.size(0)

                _, preds = torch.max(outputs, 1)
                val_acc_metric.update(preds, labels)

        if val_sample_count == 0:
            avg_val_loss = float('inf')
        else:
            avg_val_loss = val_loss_sum / val_sample_count

        current_val_accuracy = val_acc_metric.compute().item()

        if early_stopper(avg_val_loss):
            print(f"Epoch {epoch+1}: Early stopping triggered")

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
    
    return -best_val_accuracy

def cnn_objective(hyperparameters):
    # f1_channels, f2_channels, kernel_size_1, kernel_size_2, activation_str, lr, batch_size, num_epochs = hyperparameters

    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    best_val_accuracy = -np.inf

    np.random.seed(42)

    data_tr, data_val, _, labels_tr, labels_val, _ = split_data(data_root, train_ratio=0.65, val_ratio=0.15)

    # Load data
    transform = transforms.ToTensor()
    train_loader = DataLoader(CustomImageDataset(data_tr, labels_tr, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CustomImageDataset(data_val, labels_val, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=False)

    # set up for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if device == "cuda":
            return train_and_evaluate(hyperparameters, train_loader, val_loader, device)
        else:
            # Skip the try block if CUDA is not available at all
            raise RuntimeError("CUDA is not available, forcing CPU run.")
    
    except RuntimeError as e:
        # Catch any RuntimeError (which includes CUDA/CUBLAS/OOM/Max-Pool errors)
        # If the error is likely GPU-related, switch to CPU.
        print("\n" + "="*80)
        print(f"!!! WARNING: GPU-related RuntimeError encountered: {e}")
        print("!!! Switching to CPU and rerunning training attempt...")
        print("="*80 + "\n")
        
        # Attempt 2: Rerun on CPU
        device = "cpu"
        # Clear CUDA cache just in case the error was OOM before switching
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        # We re-raise the exception if the model fails again on CPU (unlikely)
        return train_and_evaluate(hyperparameters, train_loader, val_loader, device)


space = [
    Integer(10, 100, name='neurons'),
    Categorical(['relu', 'sigmoid', 'tanh'], name='activation'),
    Integer(1, 3, name='layers1'),
    Integer(1, 3, name='layers2'),
    Integer(3, 5, name='kernel_size'),
    # Integer(3, 5, name='kernel_size_2'),
    Real(0, 0.5, name='dropout_rate'),
    Categorical([0, 1], name='normalization'),
    Real(1e-5, 1e-2, 'log-uniform', name='lr'),
    Integer(32, 128, name='batch_size'),
    Integer(5, 20, name='num_epochs')
]

def final_test_run(hyperparameters, run_seed):
    
    # 1. Setup Seeds and Hyperparameters
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    # Ensure filters, batch_size, and epochs are integers
    neurons, activation_str, layers1, layers2, kernel_size, dropout_rate, normalization, lr, batch_size, num_epochs = hyperparameters

    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    layers1 = int(layers1)
    layers2 = int(layers2)
    neurons = int(neurons)
    kernel_size = int(kernel_size)
    
    # 2. Data Split (New split for each run)
    # Ratio: 70% Train, 15% Val, 15% Test. Train and Val are combined for final training.
    data_tr, data_val, data_t, labels_tr, labels_val, labels_t = split_data(
        data_root, train_ratio=0.65, val_ratio=0.15 
    )
    # Combine Train and Val for the final, full training set
    data_tr_final = data_tr + data_val
    labels_tr_final = labels_tr + labels_val

    transform = transforms.ToTensor()
    train_loader = DataLoader(CustomImageDataset(data_tr_final, labels_tr_final, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CustomImageDataset(data_t, labels_t, transform=transform), batch_size=batch_size, shuffle=False)
    
    # 3. Model Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda":
            model = CNN(neurons=neurons, in_channels=3, activation_fn_str=activation_str, layers1=layers1, layers2=layers2, kernel_size_1=kernel_size, kernel_size_2=kernel_size, dropout_rate=dropout_rate, normalization=normalization, num_classes=num_classes, img_h=IMG_HEIGHT, img_w=IMG_WIDTH).to(device)
            print(f"Model initialized on {device}")
        else:
            # Skip the try block if CUDA is not available at all
            raise RuntimeError("CUDA is not available, forcing CPU run.")
    except RuntimeError as e:
        # Catch any RuntimeError (which includes CUDA/CUBLAS/OOM/Max-Pool errors)
        # If the error is likely GPU-related, switch to CPU.
        print("\n" + "="*80)
        print(f"!!! WARNING: GPU-related RuntimeError encountered: {e}")
        print("!!! Switching to CPU and rerunning training attempt...")
        print("="*80 + "\n")
        
        # Attempt 2: Rerun on CPU
        device = "cpu"
        # Clear CUDA cache just in case the error was OOM before switching
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        # We re-raise the exception if the model fails again on CPU (unlikely)
        return train_and_evaluate(hyperparameters, train_loader, val_loader, device)

    # 4. Training (on Train + Val data)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 5. Evaluation on Test Set
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    prec_metric = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    rec_metric = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            acc_metric.update(preds, labels)
            prec_metric.update(preds, labels)
            rec_metric.update(preds, labels)

    return (
        acc_metric.compute().item(),
        prec_metric.compute().item(),
        rec_metric.compute().item()
    )

def baye():
    res_gp = None
    if os.path.exists(OPTMIZED_HPS_PATH):
        print(f"Loading optimized hyperparameters from {OPTIMIZED_HPS_PATH}")
        with open(OPTIMIZED_HPS_PATH, 'rb') as f:
            res_gp = skopt.load(f)
    else:
        print("Running Bayesian Optimization...")
        res_gp = gp_minimize(
            cnn_objective,   # Function to minimize (returns -Validation Accuracy)
            space,           # Hyperparameter search space
            n_calls=30,      # Total number of function evaluations (e.g., 30 experiments)
            n_random_starts=10, # Number of random points to start with
            random_state=42  # Seed for reproducibility of the BO process
        )
        skopt.dump(res_gp, OPTIMIZED_HPS_PATH)
        print(f"Optimization results saved to {OPTIMIZED_HPS_PATH}")

        best_hps = res_gp.x

        print(f"Best hyperparameters is {best_hps}")
    


def train():

    res_gp = None
    if os.path.exists(OPTIMIZED_HPS_PATH):
        print(f"Loading optimized hyperparameters from {OPTIMIZED_HPS_PATH}")
        with open(OPTIMIZED_HPS_PATH, 'rb') as f:
            res_gp = skopt.load(f)
    else:
        print("Running Bayesian Optimization...")
        res_gp = gp_minimize(
            cnn_objective,   # Function to minimize (returns -Validation Accuracy)
            space,           # Hyperparameter search space
            n_calls=30,      # Total number of function evaluations (e.g., 30 experiments)
            n_random_starts=10, # Number of random points to start with
            random_state=42  # Seed for reproducibility of the BO process
        )
        skopt.dump(res_gp, OPTIMIZED_HPS_PATH)
        print(f"Optimization results saved to {OPTIMIZED_HPS_PATH}")
    


    best_hps = res_gp.x
    best_validation_score = -res_gp.fun

    print("Bayesian Optimization Complete.")
    print(f"Best Hyperparameters: {dict(zip([s.name for s in space], best_hps))}")

    seeds = [100, 200, 300, 400, 500]
    results = []
    print("Starting 5-Run Final Test with Best Hyperparameters...")

    # Execute 5 runs
    for i, seed in enumerate(seeds):
        print(f"--- Running Test {i+1}/5 with seed {seed} ---")
        # This will train a new model from scratch and generate a new random data split
        acc, prec, rec = final_test_run(best_hps, seed)
        results.append((acc, prec, rec))
        print(f"Results: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

    # Calculate Average and Standard Deviation
    results_array = np.array(results)

    avg_acc = np.mean(results_array[:, 0])
    std_acc = np.std(results_array[:, 0])
    avg_prec = np.mean(results_array[:, 1])
    std_prec = np.std(results_array[:, 1])
    avg_rec = np.mean(results_array[:, 2])
    std_rec = np.std(results_array[:, 2])

    print("\n" + "=" * 50)
    print("SECTION 6.1: FINAL EXPERIMENTAL RESULTS (AVERAGE OVER 5 RUNS)")
    print("=" * 50)
    print(f"Test Accuracy: {avg_acc:.4f} \u00B1 {std_acc:.4f}")
    print(f"Macro Precision: {avg_prec:.4f} \u00B1 {std_prec:.4f}")
    print(f"Macro Recall: {avg_rec:.4f} \u00B1 {std_rec:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Assignemnt 2')
    # parser.add_argument(
    #     "-t", "--train",
    #     help='Train'
    # )
    # parser.add_argument(
    #     "-b", "--baye",
    #     help='Bayesian'
    # )

    # args = parser.parse_args()
    # if args.train:
    #     train()
    # if args.baye:
    #     baye()

    train()