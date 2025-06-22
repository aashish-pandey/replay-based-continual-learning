import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


#set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#Define SplitMNIST
def get_split_mnist(task_id):
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform = transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform = transform)

    class_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    selected_digits = class_pairs[task_id]

    def filter_digits(dataset):
        idx = [i for i, (x, y) in enumerate(dataset) if y in selected_digits]
        subset = Subset(dataset, idx)
        return subset
    
    train_data = filter_digits(mnist_train)
    test_data = filter_digits(mnist_test)

    return train_data, test_data, selected_digits

#Define a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=2):
        super(SimpleMLP, self).__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.view(x.size(0), -1)  #Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
#Training Loop per task
def train_task(model, train_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(5):
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            labels = (labels == labels.unique()[1]).long()  #Map to 0/1
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

#Evaluation
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            labels = (labels == labels.unique()[1]).long()
            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


#Main Script
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    acc_per_task = []

    for task_id in range(5):
        print(f"\nTraining for task {task_id + 1}")
        train_data, test_data, digits = get_split_mnist(task_id)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size = 64)

        train_task(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        acc_per_task.append(acc)
        print(f"Accuracy on Task {task_id + 1} ({digits}): {acc:.4f}")
    print("\nTask-wise accuracy: ", acc_per_task)