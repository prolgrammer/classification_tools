import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from utils.logger import Logger

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, class_names, log_dir='logs', plot_dir='plots'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.logger = Logger(log_dir)
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            test_loss, test_acc = self.evaluate()
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            log_message = (f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                           f"Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            self.logger.log(log_message)
            print(log_message)

        self.save_model('tools_classifier.pth')
        self.plot_metrics()

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(self.test_loader)
        test_acc = 100 * correct / total
        return test_loss, test_acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def predict(self, image):
        self.model.eval()
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        return self.class_names[predicted.item()]

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()