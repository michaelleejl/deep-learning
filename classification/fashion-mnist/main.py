import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision.io import read_image
from torch import nn
# ToTensor: Convert a PIL image or np.ndarray to FloatTensor and scales intensity values within [0, 1]
from torchvision.transforms import ToTensor, Lambda

# Dataset: Primitive that allows for both preloaded and own data
from torch.utils.data import Dataset
# DataLoader: Primitive that wraps Dataset in iterable
from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(
    root="data",  # path where data is stored
    train=True,
    download=True,
    transform=ToTensor(),  # specify the feature transformation
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), src=1))
    # Lambda specifies a lambda function for transformation
    # Zeros creates a vector of length 10, and scatter_ scatters the source, 1, according to the index
    # This creates a one hot vector encoding
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


def data_visualisation():
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = (3, 3)
    for i in range(1, cols * rows + 1):
        # random integer, tensor of shape [1]
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


class CustomImageDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# DataLoader: An API that offers three key services
#   (1) Pass samples in mini batches rather than one by one
#   (2) Reshuffle samples, to reduce over-fitting. Data is reshuffled after all batches are processed.
#   (3) Use multiprocessing to speed up data retrieval
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


def iterate_dataset():
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    lbl = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(lbl)


device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
print(f"Using device {device}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # flatten flattens the input into a single dimension. dim 0 is the batch size, it is unaffected.
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # applies a linear transform using weights and biases. 28 * 28 input features and 512 output
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # if you call model(x), you are implicitly calling forward with some background functions
    # do not call forward directly
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# creates an instance of the neural network and pushes it to the device
model = NeuralNetwork()


def test_nn():
    X = torch.rand(2, 28, 28, device=device)
    logits = model(X)
    # dim: which dimension along which the values must sum to 1 (1=row)
    predicted_probability = nn.Softmax(dim=1)(logits)
    print(predicted_probability)
    y_predicted = predicted_probability.argmax(1)
    print(f"Predicted class: {y_predicted}")
    print(f"Model structure: {model}\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# HYPERPARAMETERS
epochs = 5  # number of times to iterate through dataset
batch_size = 64  # number of samples propagated through the network before parameters updated
learning_rate = 1e-3  # how much to update models on every batch/epoch. small: low training speed. large: unpredictable.

loss_function = nn.CrossEntropyLoss()
# Optimisation adjusts model parameters to reduce error at each step
# Optimisation algorithms define how this is done
# SGD - Stochastic Gradient Descent. Others: ADAM, RMSProp
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_function(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_function, optimizer)
    test_loop(test_dataloader, model, loss_function)
print("Done!")

# AFTER TRAINING
# Saving and loading the model
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))
model.eval() # Sets dropout and batch normalisation layers to evaluation mode. Otherwise, inconsistent results
# Or can also save the class alongside the weights
torch.save(model, 'model.pth')
model = torch.load('model.pth')

