import torch
from tqdm import tqdm

def test(model, device, test_loader):
    model.to(device)
    model.eval()
    correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            prediction = outputs.argmax(dim=1)
            total_samples += images.size(0)
            correct += (prediction == labels).sum().item()

    test_acc = correct / total_samples
    print(f"Test Accuracy: {test_acc:.4f}")
