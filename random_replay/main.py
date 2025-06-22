import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os


#set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#Defining replay buffer class
class ReplayBuffer:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, x, y):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((x.cpu(), y.cpu()))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(len(self.buffer), batch_size), replace=False)
        x, y = zip(*[self.buffer[i] for i in indices])
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y


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
def train_task(model, train_loader, criterion, optimizer, device, replay_buffer):
    model.train()
    for epoch in range(5):
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            labels = (labels == labels.unique()[1]).long()  #Map to 0/1

            #Add current samples to replay buffer
            for i in range(x.size(0)):
                replay_buffer.add(x[i], labels[i])
            
            #Sample from buffer and combine
            if len(replay_buffer.buffer) > 0:
                x_replay, y_replay = replay_buffer.sample(batch_size=32)
                x_replay, y_replay = x_replay.to(device), y_replay.to(device)

                #Mix replayed and current batch
                x = torch.cat([x, x_replay])
                labels = torch.cat([labels, y_replay])

            #Normal Training
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

    task_loaders = []
    acc_matrix = []

    #Folders to save plots and logs
    os.makedirs("results", exist_ok=True)

    #initializing the replay buffer
    replay_buffer = ReplayBuffer(capacity=6000) #Total MNIST train size is 60,000 and test is 10,000. We are setting the buffer at 10% of train size i.e. 6,000


    for task_id in range(5):
        print(f"\nTraining on Task {task_id + 1}")
        train_data, test_data, digits = get_split_mnist(task_id)
        train_loader = DataLoader(train_data, batch_size = 64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64)

        #save test loader for future evaluation
        task_loaders.append((test_loader, digits))

        #Train on current task
        train_task(model, train_loader, criterion, optimizer, device, replay_buffer)

        #Evaluate on all tasks learned so far
        task_accs = []
        for i, (loader, digit_pair) in enumerate(task_loaders):
            acc = evaluate(model, loader, device)
            task_accs.append(acc)
            print(f"Accuracy on Task {i+1} {digit_pair}: {acc:.4f}")

        acc_matrix.append(task_accs)
    
        print("Task accuracies: ", task_accs)

    #acc_matrix has inhomogenous shape that is each row has one more column than previous cause we train for one new task
    #Lets make it square
    max_len = max(len(row) for row in acc_matrix)
    padded_matrix = [row + [np.nan] * (max_len - len(row)) for row in acc_matrix]


    #Save accuracy matrix as csv
    np.savetxt("results/acc_matrix.csv", np.array(padded_matrix), delimiter=",")

    #Convert to numpy for easy plotting
    acc_array = np.array(padded_matrix)

    #plot: accuracy per task over time
    plt.figure(figsize=(10, 6))
    for task_id in range(acc_array.shape[1]):
        plt.plot(range(1, acc_array.shape[0]+1), acc_array[:, task_id], marker='o',  label = f"Task {task_id + 1}")
    plt.title("Task wise accuracy over time")
    plt.xlabel("Training step (Task ID)")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/accuracy_plot.png")
    plt.close()
