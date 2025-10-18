#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import cv2 as cv
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(9, 18, 5)
        self.fc1 = torch.nn.Linear(18 * 133 * 60, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch._C._VariableFunctions.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def pre_processing(): #Pre-process dataset as needed.
    images = []
    labels = []

    path = './Dhan-Shomadhan'
    count_b = -1
    count_d = -1

    for background in (os.listdir(path)):
        if(background == 'Dhan-Shomadhan_picture_Information.csv' or background == '.DS_Store'):
            continue
        count_b = count_b + 1
        print('Background:', count_b)

        for disease in (os.listdir(os.path.join(path, background))):

            if count_d>=4:
                count_d = -1

            count_d = count_d + 1
            print('Disease:', count_d)
            for img in os.listdir(os.path.join(path,background, disease)):
                a = cv.imread(os.path.join(path,background, disease, img))
                h, w = a.shape[:2]
                if h > w:
                    # Rotate 90 degrees clockwise if the Height is larger than the Width
                    a = cv.rotate(a, cv.ROTATE_90_CLOCKWISE)
                a = cv.resize(a, (544, 255)) #Use the same width to heigh ratio as original images
                images.append(a)
                labels.append((count_d))


    img_data= np.array(images)
    label_data = np.array(labels)

    img_tensor = torch.from_numpy(img_data).permute(0, 3, 1, 2).float() / 255.0
    label_tensor = torch.from_numpy(label_data)
    dataset = TensorDataset(img_tensor,label_tensor)

    generator1 = torch.Generator().manual_seed(42)
    train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2], generator1)
    torch.save(train_data, "train_dataset.pth")
    torch.save(test_data, "test_dataset.pth")


def feature_extraction(dataset_path):
    data = []

    return data

def train_cnn(train_loader):#Take in the hyperparameters as well as dataset path. Setup CNN structure and train.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(device)
    num_epochs = 10

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = net(data)
            loss = criterion(scores, targets)

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()

    torch.save(net.state_dict(), 'cnn_model.pth')
    print('success')

    return

def run_cnn(test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net().to(device)
    net.load_state_dict(torch.load('cnn_model.pth', map_location=device))
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def hyperparam_randomSearch(): #Randomsearch over hyperparameters
    return

def model_crossValidation(): #Cross validate model.
    return

def plot_results():
    return

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        action="store_true",
        help="Create test train split datasets"
    )
    parser.add_argument(
        "-f",
        action="store_true",
        help="Run training"
    )
    parser.add_argument(
        "-r",
        action="store_true",
        help="Run testing"
    )

    args = parser.parse_args()

    if args.p:
        pre_processing()

    if args.f:
        train_data = torch.load("train_dataset.pth", weights_only=False)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        train_cnn(train_loader)

    if args.r:
        test_data = torch.load("test_dataset.pth", weights_only=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
        run_cnn(test_loader)




if __name__ == "__main__":
    main()