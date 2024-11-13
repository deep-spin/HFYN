from utils import HopfieldNet, Flatten
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import argparse
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description='MNIST arguments:')
    parser.add_argument('--dataset', default='MNIST', type=str, choices=["MNIST", "CIFAR10", "ImageNet"])
    parser.add_argument('--yomega', default='10_poly', type=str, choices=["identity", "entmax", "normmax", "10_poly", "5_poly", "exp"])
    parser.add_argument('--alpha', help='alpha', default=0.1, type=float)
    parser.add_argument('--ypsi', default='tanh', type=str, choices=["none", "normalization", "layer_normalization", "tanh", "identity"])
    parser.add_argument('--beta', help='inverse of temperature', default=0.1, type=float)
    parser.add_argument('--inner_beta', help='inverse of temperature for the exp DAM inner operation', default=0.05, type=float)
    parser.add_argument('--std', help='std for noise in queries', default=0, type=float)
    parser.add_argument('--perc', help="mask percentage", default=0.5, type=float)
    parser.add_argument('--memory_size', help='number of memories', default=2, type=int)
    parser.add_argument('--device', help='gpu', default=1, type=int)
    args = parser.parse_args()
    return args


def load_MNIST():
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5),  # Normalize to [-1, 1]
    ])

    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='../datasets', train=True, download=False, transform=transform)
    mnist_dataset_test = datasets.MNIST(root='../datasets', train=False, download=False, transform=transform)

    # Create a DataLoader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=True)
    # Create a DataLoader to iterate over the dataset
    data_loader_test = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=len(mnist_dataset_test), shuffle=True)

    for data in data_loader:
        X_train, labels_train = data

    for data in data_loader_test:
        X_test, labels_test = data
    
    return X_train, X_test

def load_tiny_ImageNet():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root='../datasets/tiny-imagenet-200/train',
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                              shuffle=True, num_workers=4)

    train_data = next(iter(trainloader))

    testset = torchvision.datasets.ImageFolder(root='../datasets/tiny-imagenet-200/test',
                                               transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=4)

    test_data = next(iter(testloader))

    return train_data[0], test_data[0]

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                      shuffle=True)
    train_data = next(iter(trainloader))
    testset = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                      shuffle=True)
    test_data = next(iter(testloader))
    
    return train_data[0], test_data[0]

def add_gaussian_noise(images, mean=0, std=1):
    # Generate Gaussian noise with the same shape as the images

    noise = torch.randn(images.size()) * std + mean
    # Add noise to the images
    noisy_images = images + noise.to(images.device)
    # Ensure pixel values are within the valid range (0 to 1 for normalized images)
    noisy_images = torch.clamp(noisy_images, -1, 1)
    return noisy_images

def whiten_image(X, perc, args):
    # Define the image dimensions based on the dataset
    if args.dataset == "MNIST":
        rows, cols = 28, 28
    elif args.dataset == "CIFAR10":
        rows, cols = 32, 32
    elif args.dataset == "ImageNet":
        rows, cols = 64, 64
    else:
        raise ValueError("Unsupported dataset")

    # Calculate the number of rows to whiten
    rows_to_whiten = int(perc * rows)

    # Clone the input tensor to avoid in-place modification
    X_whitened = X.clone()

    # Perform the whitening operation
    if rows_to_whiten != 0:
        X_whitened[:, :, -rows_to_whiten:, :] = 0

    # Reshape the tensor as required
    X_whitened = X_whitened.reshape(X_whitened.shape[0], -1)

    return X_whitened

def Hopfield(args):
    torch.random.manual_seed(42)
    
    if args.dataset == "MNIST":        
        X, _ = load_MNIST()
    
    elif args.dataset == "CIFAR10":
        X,_ = load_cifar10()
    
    elif args.dataset == "ImageNet":
        X,_ = load_tiny_ImageNet()

    sims = []
    for i in range(5):
        torch.manual_seed(i)

        # Shuffle the dataset
        all_indices = torch.randperm(X.shape[0])

        # Select the first 10 indices
        indices = all_indices[:args.memory_size]
        X_stored = X[indices].to(args.device)
        Q = add_gaussian_noise(X_stored, 0, args.std)
        Q_noisy = whiten_image(Q, args.perc, args)
        X_stored = X_stored.reshape(args.memory_size, -1)
        model = HopfieldNet(X_stored, args.yomega, args.alpha, args.ypsi, args.beta, args.inner_beta, 5, args.device)
        Q = model(Q_noisy.T.to(args.device), False).clone()

        if args.ypsi == "normalization" or args.ypsi == "layer_normalization":
            X_stored = getattr(model, f"_{args.ypsi}", None)(tensor=X_stored, axis=1)

        cosine_sims = F.cosine_similarity(X_stored, Q.T, dim=1)
        # Count the number of values higher than 0.9
        count_high_values = (cosine_sims > 0.9).sum().item()

        # Calculate the proportion by dividing by the total number of elements
        sims.append(count_high_values / cosine_sims.numel())
    
    sim = torch.tensor(sims)
    print(f"Mean: {torch.mean(sim).item()}, Standard Deviation: {torch.std(sim).item()}")
    
    return sims

if __name__ == '__main__':
    args = get_args()
    Hopfield(args)