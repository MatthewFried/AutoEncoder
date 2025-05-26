import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch
import urllib.request
import zipfile
import os

# Set random seed for reproducibility
np.random.seed(42)

# --- Dataset Loading and Preprocessing ---
def load_fashion_mnist():
    """Load and preprocess Fashion-MNIST dataset, saving to .npy files."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy().astype('float32') / 255.0
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy().astype('float32') / 255.0
    y_test = test_dataset.targets.numpy()

    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # Save to .npy files
    np.save('fashion_mnist_x_train.npy', x_train)
    np.save('fashion_mnist_y_train.npy', y_train)
    np.save('fashion_mnist_x_test.npy', x_test)
    np.save('fashion_mnist_y_test.npy', y_test)
    print("Fashion-MNIST data saved to .npy files.")

def load_diabetes():
    """Load and preprocess UCI Diabetes dataset, saving to .npy files."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    data = pd.read_csv(url, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save to .npy files
    np.save('diabetes_x_train.npy', x_train)
    np.save('diabetes_y_train.npy', y_train)
    np.save('diabetes_x_test.npy', x_test)
    np.save('diabetes_y_test.npy', y_test)
    print("UCI Diabetes data saved to .npy files.")

def load_mnist():
    """Load and preprocess MNIST dataset, saving to .npy files."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy().astype('float32') / 255.0
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy().astype('float32') / 255.0
    y_test = test_dataset.targets.numpy()

    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # Save to .npy files
    np.save('mnist_x_train.npy', x_train)
    np.save('mnist_y_train.npy', y_train)
    np.save('mnist_x_test.npy', x_test)
    np.save('mnist_y_test.npy', y_test)
    print("MNIST data saved to .npy files.")

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset, saving to .npy files."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Convert to numpy and normalize
    x_train = np.array(train_dataset.data, dtype=np.float32) / 255.0
    y_train = np.array(train_dataset.targets, dtype=np.int64)
    x_test = np.array(test_dataset.data, dtype=np.float32) / 255.0
    y_test = np.array(test_dataset.targets, dtype=np.int64)

    # Reshape: (N, H, W, C) -> (N, C, H, W) -> (N, C*H*W)
    x_train = x_train.transpose(0, 3, 1, 2).reshape(-1, 3 * 32 * 32)
    x_test = x_test.transpose(0, 3, 1, 2).reshape(-1, 3 * 32 * 32)

    # Save to .npy files
    np.save('cifar10_x_train.npy', x_train)
    np.save('cifar10_y_train.npy', y_train)
    np.save('cifar10_x_test.npy', x_test)
    np.save('cifar10_y_test.npy', y_test)
    print("CIFAR-10 data saved to .npy files.")

def load_dsprites():
    """Load and preprocess dSprites dataset, saving to .npy files with ground truth factors."""
    # Download dSprites dataset
    dsprites_url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    dsprites_path = "./data/dsprites.npz"
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists(dsprites_path):
        urllib.request.urlretrieve(dsprites_url, dsprites_path)

    # Load the .npz file
    data = np.load(dsprites_path)
    images = data['imgs']  # Shape: (737280, 64, 64)
    latents = data['latents_values']  # Shape: (737280, 6), includes 'label' (first column)

    # Subsample to 50,000 samples
    subset_size = 50000
    indices = np.random.choice(len(images), subset_size, replace=False)
    images = images[indices]
    latents = latents[indices][:, 1:]  # Exclude 'label' (first column), keep shape, scale, rotation, x, y

    # Convert to float32 and flatten images
    x_data = images.astype(np.float32).reshape(-1, 64 * 64)  # Shape: (50000, 4096)
    y_data = latents.astype(np.float32)  # Shape: (50000, 5)

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Save to .npy files
    np.save('dsprites_x_train.npy', x_train)
    np.save('dsprites_y_train.npy', y_train)
    np.save('dsprites_x_test.npy', x_test)
    np.save('dsprites_y_test.npy', y_test)
    print("dSprites data saved to .npy files.")

def load_wine():
    """Load and preprocess UCI Wine dataset, saving to .npy files."""
    from sklearn.datasets import load_wine

    # Load UCI Wine
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save to .npy files
    np.save('wine_x_train.npy', x_train)
    np.save('wine_y_train.npy', y_train)
    np.save('wine_x_test.npy', x_test)
    np.save('wine_y_test.npy', y_test)
    print("UCI Wine data saved to .npy files.")

def load_celeba():
    """Load and preprocess CelebA dataset, saving to .npy files with attributes."""
    # Define transform: resize to 64x64 and flatten
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to (N, 3*64*64)
    ])

    # Load CelebA
    train_dataset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    test_dataset = datasets.CelebA(root='./data', split='test', download=True, transform=transform)

    # Subsample to 10,000 training and 2,000 test samples
    train_subset_size = 10000
    test_subset_size = 2000

    train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)

    # Extract data and attributes
    x_train = np.array([train_dataset[i][0].numpy() for i in train_indices], dtype=np.float32)
    y_train = np.array([train_dataset[i][1].numpy() for i in train_indices], dtype=np.float32)  # Attributes
    x_test = np.array([test_dataset[i][0].numpy() for i in test_indices], dtype=np.float32)
    y_test = np.array([test_dataset[i][1].numpy() for i in test_indices], dtype=np.float32)

    # Save to .npy files
    np.save('celeba_x_train.npy', x_train)
    np.save('celeba_y_train.npy', y_train)
    np.save('celeba_x_test.npy', x_test)
    np.save('celeba_y_test.npy', y_test)
    print("CelebA data saved to .npy files.")

if __name__ == "__main__":
    load_fashion_mnist()
    load_diabetes()
    load_mnist()
    load_cifar10()
    load_dsprites()
    load_wine()