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
OPTIMIZED_HPS_PATH_PRETRAINED = 'optimized_hps_pretrained.pkl'

pretrained = True
default = False

# You could calculate your dataset's specific mean/std for better results.
MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]

# train_transforms = transforms.Compose([
#     transforms.ToTensor(),             # Converts image to tensor and scales to [0, 1]
#     transforms.RandomHorizontalFlip(p=0.5), # Augmentation 1: Random flip
#     transforms.RandomRotation(degrees=15),  # Augmentation 2: Small random rotation
#     transforms.ColorJitter(brightness=0.1, contrast=0.1), # Augmentation 3: Minor color variation
#     transforms.Normalize(MEAN, STD)    
# ])

train_transforms = transforms.Compose([
    transforms.ToTensor(),             
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.2), # Added: More generalizable flips
    transforms.RandomRotation(degrees=30),  # Increased rotation
    # Increased Jitter to force robustness against lighting/shadows (Field background)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), 
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

# --- CNN Model (Kept the same) ---
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
        cur_channel = neurons
        for _ in range(layers1):
            layers_1_list.append(nn.Conv2d(cur_channel, cur_channel, kernel_size=kernel_size_1, padding='same')) # Use 'same' for simplicity
            layers_1_list.append(self.activation)
            layers_1_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block1 = nn.Sequential(*layers_1_list)
        
        # Block 2 (Repeated Conv-Pool) - Use different kernel size if desired
        layers_2_list = []
        if dropout_rate > 0:
            layers_2_list.append(nn.Dropout2d(p=dropout_rate))

        # cur_channel remains the same from block1's last output
        for _ in range(layers2):
            layers_2_list.append(nn.Conv2d(cur_channel, cur_channel * 2, kernel_size=kernel_size_2, padding='same')) # Double filters for deeper layers
            layers_2_list.append(self.activation)
            layers_2_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            cur_channel = cur_channel * 2

        self.block2 = nn.Sequential(*layers_2_list)
        
        dummy_input = torch.zeros(1, in_channels, img_h, img_w) 
        
        with torch.no_grad():
            x = self.initial_block(dummy_input)
            x = self.block1(x)
            x = self.block2(x)
            
        fc_input_size = x.numel() // x.size(0)
        
        self.fc1 = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.initial_block(x)
        x = self.block1(x)
        x = self.block2(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
    
def create_model(hyperparameters):
    if not pretrained:
        neurons, activation_str, layers1, layers2, kernel_size, dropout_rate, normalization, lr, batch_size, num_epochs = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)
        layers1 = int(layers1)
        layers2 = int(layers2)
        neurons = int(neurons)
        kernel_size = int(kernel_size)
        model = CNN(neurons=neurons, in_channels=3, activation_fn_str=activation_str, layers1=layers1, layers2=layers2, kernel_size_1=kernel_size, kernel_size_2=kernel_size, dropout_rate=dropout_rate, normalization=normalization, num_classes=num_classes, img_h=IMG_HEIGHT, img_w=IMG_WIDTH)
    else:
        lr, lr_backbone, batch_size, num_epochs, weight_decay = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the FINAL two blocks (layer4 and layer3)
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True

        # 3. Replace and Unfreeze the final FC layer (classification head)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    return model

# --- train_and_evaluate (Kept the same) ---
def train_and_evaluate(hyperparameters, train_loader, val_loader, device):
    if not pretrained:
        neurons, activation_str, layers1, layers2, kernel_size, dropout_rate, normalization, lr, batch_size, num_epochs = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)
        layers1 = int(layers1)
        layers2 = int(layers2)
        neurons = int(neurons)
        kernel_size = int(kernel_size)
    else:
        lr, lr_backbone, batch_size, num_epochs, weight_decay = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)
    
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    best_val_accuracy = -np.inf

    model = create_model(hyperparameters)
    print(model)

    criterion = nn.CrossEntropyLoss()
    if pretrained:
        optimizer = optim.Adam([
            {'params': model.fc.parameters(), 'lr': lr, 'weight_decay': 1e-4}, # High L.R. for the new layer
            {'params': model.layer4.parameters(), 'lr': lr_backbone, 'weight_decay': weight_decay}, # Low L.R. for fine-tuning
            {'params': model.layer3.parameters(), 'lr': lr_backbone, 'weight_decay': weight_decay}, # Low L.R. for fine-tuning
        ], lr=lr_backbone) # Default L.R. for any other parameters (will be ignored if frozen)
    else:
        # Use the single L.R. from the optimization space for custom CNN
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
        print(f"Epoch {epoch+1}: Validation loss {avg_val_loss}, Validation accuracy {current_val_accuracy}")


        if early_stopper(avg_val_loss):
            print(f"Epoch {epoch+1}: Early stopping triggered")
            break;

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
    
    return -best_val_accuracy

# --- cnn_objective (Updated to use new transforms) ---
def cnn_objective(hyperparameters):
    if not pretrained:
        neurons, activation_str, layers1, layers2, kernel_size, dropout_rate, normalization, lr, batch_size, num_epochs = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)
        layers1 = int(layers1)
        layers2 = int(layers2)
        neurons = int(neurons)
        kernel_size = int(kernel_size)
    else:
        lr, lr_backbone, batch_size, num_epochs, weight_decay = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)

    np.random.seed(42)

    data_tr, data_val, _, labels_tr, labels_val, _ = split_data(data_root, train_ratio=0.7, val_ratio=0.15)

    train_loader = DataLoader(CustomImageDataset(data_tr, labels_tr, transform=train_transforms), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CustomImageDataset(data_val, labels_val, transform=test_val_transforms), batch_size=batch_size, shuffle=False)

    # set up for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if device == "cuda":
            return train_and_evaluate(hyperparameters, train_loader, val_loader, device)
        else:
            raise RuntimeError("CUDA is not available, forcing CPU run.")
    
    except RuntimeError as e:
        print("\n" + "="*80)
        print(f"!!! WARNING: GPU-related RuntimeError encountered: {e}")
        print("!!! Switching to CPU and rerunning training attempt...")
        print("="*80 + "\n")
        
        device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        return train_and_evaluate(hyperparameters, train_loader, val_loader, device)

space = [
    Integer(10, 100, name='neurons'),
    Categorical(['relu', 'sigmoid', 'tanh'], name='activation'),
    Integer(1, 3, name='layers1'),
    Integer(1, 3, name='layers2'),
    Integer(3, 5, name='kernel_size'),
    Real(0, 0.5, name='dropout_rate'),
    Categorical([0, 1], name='normalization'),
    Real(1e-5, 1e-2, 'log-uniform', name='lr'),
    Integer(32, 128, name='batch_size'),
    Integer(10, 30, name='num_epochs')
]

space_pretrained =[
    Real(1e-4, 1e-2, 'log-uniform', name='lr'),
    Real(1e-6, 1e-4, 'log-uniform', name='lr_backbone'),    # Very low for fine-tuning layers 3 and 4
    Integer(32, 128, name='batch_size'),
    Integer(20, 40, name='num_epochs'),
    Real(1e-5, 1e-3, 'log-uniform', name='weight_decay')
]

default_hp_cnn = [32, 'relu', 1, 1, 3, 0.0, 1, 1e-4, 64, 20]
default_hp_pretrained = [1e-4, 32, 30]


def final_test_run(hyperparameters, seed):

    if not pretrained:
        neurons, activation_str, layers1, layers2, kernel_size, dropout_rate, normalization, lr, batch_size, num_epochs = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)
        layers1 = int(layers1)
        layers2 = int(layers2)
        neurons = int(neurons)
        kernel_size = int(kernel_size)
    else:
        lr, lr_backbone, batch_size, num_epochs, weight_decay = hyperparameters
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)

    model = create_model(hyperparameters)
    print(model)
    
    # 1. Setup Seeds and Hyperparameters
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 2. Data Split (New split for each run)
    data_tr, data_val, data_t, labels_tr, labels_val, labels_t = split_data(
        data_root, train_ratio=0.7, val_ratio=0.15 
    )

    # Combine Train and Val for the final, full training set
    data_tr_final = data_tr + data_val
    labels_tr_final = labels_tr + labels_val

    print(f"lerning rate {lr}, batch size {batch_size}, number of epochs {num_epochs}")

    train_loader = DataLoader(CustomImageDataset(data_tr_final, labels_tr_final, transform=train_transforms), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CustomImageDataset(data_t, labels_t, transform=test_val_transforms), batch_size=batch_size, shuffle=False)
    
    # 3. Model Setup (Kept the same)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda":
            model = model.to(device)
            print(f"Model initialized on {device}")
        else:
            raise RuntimeError("CUDA is not available, forcing CPU run.")
    except RuntimeError as e:
        print("\n" + "="*80)
        print(f"!!! WARNING: GPU-related RuntimeError encountered: {e}")
        print("!!! Switching to CPU and rerunning training attempt...")
        print("="*80 + "\n")
        
        device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        model = CNN(neurons=neurons, in_channels=3, activation_fn_str=activation_str, layers1=layers1, layers2=layers2, kernel_size_1=kernel_size, kernel_size_2=kernel_size, dropout_rate=dropout_rate, normalization=normalization, num_classes=num_classes, img_h=IMG_HEIGHT, img_w=IMG_WIDTH).to(device)
        print(f"Model initialized on {device}")


    criterion = nn.CrossEntropyLoss()
    if pretrained:
        optimizer = optim.Adam([
            {'params': model.fc.parameters(), 'lr': lr, 'weight_decay': 1e-4}, # High L.R. for the new layer
            {'params': model.layer4.parameters(), 'lr': lr_backbone, 'weight_decay': weight_decay}, # Low L.R. for fine-tuning
            {'params': model.layer3.parameters(), 'lr': lr_backbone, 'weight_decay': weight_decay}, # Low L.R. for fine-tuning
        ], lr=lr_backbone) # Default L.R. for any other parameters (will be ignored if frozen)
    else:
        # Use the single L.R. from the optimization space for custom CNN
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
            
        print(f"Epoch {epoch+1}: Training loss {loss.item()}")

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
    best_hps = default_hp_cnn if not pretrained else default_hp_pretrained
    if not default:
        res_gp = None
        save_path = OPTIMIZED_HPS_PATH_PRETRAINED if pretrained else OPTIMIZED_HPS_PATH

        if os.path.exists(save_path):
            print(f"Loading optimized hyperparameters from {save_path}")
            with open(save_path, 'rb') as f:
                res_gp = skopt.load(f)
        else:
            print("Running Bayesian Optimization...")
            res_gp = gp_minimize(
                cnn_objective,   # Function to minimize (returns -Validation Accuracy)
                space_pretrained if pretrained else space,           # Hyperparameter search space
                n_calls=30,      # Total number of function evaluations (e.g., 30 experiments)
                n_random_starts=10, # Number of random points to start with
                random_state=42  # Seed for reproducibility of the BO process
            )
            skopt.dump(res_gp, save_path)
            print(f"Optimization results saved to {save_path}")
        best_hps = res_gp.x   
    print(f"Model{'ResNet18' if pretrained else 'CNN'} with {'default' if default else 'optimized'} hyperparameters")  
    print(f"Hyperparameters: {dict(zip([s.name for s in space], best_hps))}")
def train():
    best_hps = default_hp_cnn if not pretrained else default_hp_pretrained
    if not default:
        res_gp = None
        save_path = OPTIMIZED_HPS_PATH_PRETRAINED if pretrained else OPTIMIZED_HPS_PATH

        if os.path.exists(save_path):
            print(f"Loading optimized hyperparameters from {save_path}")
            with open(save_path, 'rb') as f:
                res_gp = skopt.load(f)
        else:
            print("Running Bayesian Optimization...")
            res_gp = gp_minimize(
                cnn_objective,   # Function to minimize (returns -Validation Accuracy)
                space_pretrained if pretrained else space,           # Hyperparameter search space
                n_calls=30,      # Total number of function evaluations (e.g., 30 experiments)
                n_random_starts=10, # Number of random points to start with
                random_state=42  # Seed for reproducibility of the BO process
            )
            skopt.dump(res_gp, save_path)
            print(f"Optimization results saved to {save_path}")
        best_hps = res_gp.x        

    print("Bayesian Optimization Complete.")
    print(f"Best Hyperparameters: {dict(zip([s.name for s in space], best_hps))}")

    print(f"Running model {'ResNet18' if pretrained else 'CNN'} with {'default' if default else 'optimized'} hyperparameters")

    seeds = [100, 200, 300, 400, 500]
    results = []
    print("Starting 5-Run Final Test with Best Hyperparameters...")

    # Execute 5 runs
    for i, seed in enumerate(seeds):
        print(f"--- Running Test {i+1}/5 with seed {seed} ---")
        acc, prec, rec = final_test_run(hyperparameters=best_hps, seed=seed)
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-pretrained', action='store_true', help='Set this flag to disable the pretrained model (i.e., set pretrained=False).')

    parser.add_argument('--default', action='store_true', help='Enable default settings.')
    parser.add_argument('--baye', action='store_true', help='Run Bayesian Optimization.')
    
    parser.add_argument('--manual', type=int, default=20)
    
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    default = args.default
    
    # For demonstration:
    print(f"Final calculated 'pretrained' status: {pretrained}")
    print(f"Final calculated 'default' status: {default}")
    
    if not pretrained:
        print("Setting global 'pretrained' to False.")
  
        pass

    if default:
        print("Enabling default settings.")
        # Perform actions for default settings
        pass

    if args.manual != 20: # Only apply if manual was changed from default
        print(f"Applying manual value: {args.manual}")
        default_hp_cnn[9] = args.manual
        default_hp_pretrained[2] = args.manual
    
    # ------------------------------------------------------------------
    # Execution Logic
    
    if args.baye:
        baye()
    else:
        train()