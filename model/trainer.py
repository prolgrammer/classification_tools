import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Для красивого прогресс-бара
from datetime import datetime
from utils.logger import Logger


class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, class_names, log_dir='logs', plot_dir='plots'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        self.logger = Logger(log_dir)
        self.plot_dir = plot_dir
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        # Логирование в файл и консоль
        self.logger.log("Инициализация ModelTrainer")
        self.logger.log(f"Устройство: {self.device}")
        self.logger.log(f"Количество классов: {len(class_names)} ({class_names})")
        self.logger.log(f"Размер train_loader: {len(train_loader)} батчей")
        self.logger.log(f"Размер test_loader: {len(test_loader)} батчей")

    def train(self, epochs=10):
        self.logger.log("\nНачало обучения")
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Прогресс-бар для эпохи
            with tqdm(self.train_loader, desc=f"Эпоха {epoch + 1}/{epochs}", unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.logger.log(f"Метки в батче: {labels.tolist()}")
                    if labels.max() >= len(self.class_names):
                        self.logger.log(f"Ошибка: метка {labels.max()} >= {len(self.class_names)}")
                        self.logger.log(f"Проблемные метки: {labels[labels >= len(self.class_names)]}")
                        raise ValueError("Некорректные метки!")

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    pbar.set_postfix({
                        'loss': f"{running_loss / (pbar.n + 1):.4f}",
                        'acc': f"{100 * correct / total:.2f}%"
                    })

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct / total
            test_loss, test_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            self.logger.log(
                f"Эпоха {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )

        self.save_model('tools_classifier.pth')
        self.plot_metrics()
        self.logger.log("Обучение завершено")

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

        return running_loss / len(self.test_loader), 100 * correct / total

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.log(f"Модель сохранена в {path}")

    def load_model(self, path):
        """Загружает веса модели из указанного файла."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель не найдена по пути: {path}")
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()
        self.logger.log(f"Модель загружена из {path}")

    def predict(self, img_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)  # Применяем softmax для получения вероятностей
            predicted_class = torch.argmax(probabilities, dim=1)
            return self.class_names[predicted_class[0]], probabilities  # Возвращаем имя класса и вероятности

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        plt.suptitle("Метрики обучения")

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Эпоха')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plot_path = os.path.join(self.plot_dir, 'training_metrics.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.log(f"Графики сохранены в {plot_path}")