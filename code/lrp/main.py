import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
import argparse
from utils import compute_lrp, plot_relevance, set_seed
from train import train
from test import test
from model import SimpleCNN

class SUN397Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['img']
        label = example['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and evaluate models for LRP')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_from_scratch', type=int, default=0, help='Train from sratch or load pre-trained model and test')
    parser.add_argument('--plot_relevance', type=int, default=0, help='Plot relevance')
    parser.add_argument('--dataset_name', type=str, default='uoft-cs/cifar10', help='Huggingface dataset name')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sample_image', type=int, default=0, help='Sample image')

    args = parser.parse_args()
    set_seed(args.seed)

    dataset_name = args.dataset_name
    train_dataset = load_dataset(dataset_name, split='train')
    train_dataset, val_dataset = train_dataset.train_test_split(test_size=args.val_size, seed=args.seed).values()
    test_dataset = load_dataset(dataset_name, split='test')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    train_dataset = SUN397Dataset(train_dataset, transform=transform)
    test_dataset = SUN397Dataset(test_dataset, transform=transform)
    val_dataset = SUN397Dataset(val_dataset, transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN()

    if args.train_from_scratch == 1:
        print("Training from scratch")
        # train
        model = train(model, device, train_loader, val_loader, epochs=args.num_epochs, lr=args.lr)

        # save model
        torch.save(model.state_dict(), 'code/lrp/model.pth')
    elif args.train_from_scratch == 0:
        print("Loading pre-trained model")
        # load model
        model = SimpleCNN()
        model.load_state_dict(torch.load('code/lrp/model.pth', weights_only=True))

        # test
        test(model, device, test_loader)

    if args.plot_relevance == 1:
        print("Plotting relevance")

        # Load a sample image
        sample_image_idx = args.sample_image
        image, label = test_dataset[sample_image_idx]

        image_rescaled = image.permute(1, 2, 0).numpy()
        image_rescaled = (image_rescaled - image_rescaled.min()) / (image_rescaled.max() - image_rescaled.min())

        image = image.unsqueeze(0).to(device)
        # Compute relevance using LRP
        relevance = compute_lrp(model, image)
        plot_relevance(image_rescaled, relevance, sample_image_idx)