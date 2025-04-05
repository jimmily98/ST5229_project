import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from datasets import load_dataset
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler



# Training
def train(model, device, train_loader, val_loader, epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_samples = 0
        correct = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_samples += images.size(0)
            train_loss += loss.item() * images.size(0)
            prediction = outputs.argmax(dim=1)
            correct += (prediction == labels).sum().item()

        scheduler.step()

        train_acc = correct / total_samples
        train_loss = train_loss / total_samples

        # Validation phase
        model.eval()
        correct, total_samples, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                prediction = outputs.argmax(dim=1)
                total_samples += images.size(0)
                correct += (prediction == labels).sum().item()

        val_loss = val_loss / total_samples
        val_acc = correct / total_samples

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]}")

    return model
