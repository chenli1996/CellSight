import torch
import matplotlib.pyplot as plt
import numpy as np

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'lr': [], 'loss': []}

    def range_test(self, data_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        self.optimizer.param_groups[0]['lr'] = start_lr
        best_loss = float('inf')

        for i, (inputs, labels) in enumerate(data_loader):
            if i >= num_iter:
                break

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            self.history['lr'].append(lrs[i])
            self.history['loss'].append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
            if loss.item() > 4 * best_loss:
                break

            self.optimizer.param_groups[0]['lr'] = lrs[i]

    def plot(self):
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()

# Example usage:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a simple model for demonstration
model = torch.nn.Linear(10, 1).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

# Create a dummy dataset
data_loader = [(torch.randn(64, 10), torch.randn(64, 1)) for _ in range(100)]

# Initialize LRFinder
lr_finder = LRFinder(model, optimizer, criterion, device)

# Run learning rate range test
lr_finder.range_test(data_loader, start_lr=1e-7, end_lr=10, num_iter=100)

# Plot the results
lr_finder.plot()
