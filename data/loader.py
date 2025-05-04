import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Читаем CSV и нормализуем метки
        self.annotations = pd.read_csv(csv_file)
        self.annotations['Label'] = self.annotations['Label'].str.lower().str.replace('screw driver', 'screwdriver')
        # Удаляем дубликаты, оставляя первую метку для каждого изображения
        self.annotations = self.annotations.drop_duplicates(subset=['Id'], keep='first')
        initial_count = len(self.annotations)
        self.image_files = [
            f for f in self.annotations['Id']
            if os.path.exists(os.path.join(image_dir, f))
        ]
        filtered_count = len(self.annotations)
        print(f"Filtered {initial_count - filtered_count} missing files. Kept {filtered_count} valid entries.")
        self.labels = self.annotations['Label'].tolist()
        self.class_names = sorted(list(set(self.labels)))
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.label_to_idx[self.labels[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label

class DatasetLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        # Папки датасета
        train_dir = os.path.join(self.data_dir, 'train_data', 'train_data')
        val_dir = os.path.join(self.data_dir, 'validation_data_V2', 'validation_data_V2')
        test_dir = os.path.join(self.data_dir, 'test_data', 'test_data')
        csv_file = os.path.join(self.data_dir, 'Annotated.csv')

        # Проверка существования папок и файла
        for path in [train_dir, val_dir, test_dir, csv_file]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} not found. Check dataset structure. Current working directory: {os.getcwd()}")

        # Загрузка тренировочного и валидационного датасетов
        train_dataset = datasets.ImageFolder(train_dir, transform=self.transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=self.test_transform)
        test_dataset = TestDataset(test_dir, csv_file, transform=self.test_transform)

        # Нормализация имен классов
        train_classes = [c.lower().replace('screw driver', 'screwdriver').replace('tool box', 'toolbox') for c in train_dataset.classes]
        val_classes = [c.lower().replace('screw driver', 'screwdriver').replace('tool box', 'toolbox') for c in val_dataset.classes]
        test_classes = [c.lower().replace('screw driver', 'screwdriver').replace('tool box', 'toolbox') for c in test_dataset.class_names]

        # Исключаем нежелательные классы
        excluded_classes = ['pebbel', 'train_data', 'validation_data_V2']
        train_classes = [c for c in train_classes if c not in excluded_classes]
        val_classes = [c for c in val_classes if c not in excluded_classes]
        test_classes = [c for c in test_classes if c not in excluded_classes]

        # Проверка согласованности классов
        all_classes = sorted(list(set(train_classes + val_classes + test_classes)))
        if not all_classes:
            raise ValueError("No valid classes found after normalization. Check dataset class names.")

        if set(train_classes) != set(val_classes) or set(train_classes) != set(test_classes):
            print("Warning: Class sets differ between datasets.")
            print(f"Train classes: {train_classes}")
            print(f"Validation classes: {val_classes}")
            print(f"Test classes: {test_classes}")
            print("Using all unique classes:", all_classes)

        # Приведение классов к единому списку
        self.class_names = all_classes
        train_dataset.classes = all_classes
        val_dataset.classes = all_classes
        test_dataset.class_names = all_classes

        # Обновление маппинга меток
        train_dataset.class_to_idx = {name: idx for idx, name in enumerate(all_classes)}
        val_dataset.class_to_idx = {name: idx for idx, name in enumerate(all_classes)}
        test_dataset.label_to_idx = {name: idx for idx, name in enumerate(all_classes)}

        # DataLoader'ы
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # После загрузки данных (в load_data())
        print("Unique labels in train:", set(y for _, y in train_dataset))
        print("Unique labels in val:", set(y for _, y in val_dataset))
        print("Unique labels in test:", set(y for _, y in test_dataset))
        print("Class names:", self.class_names)

        return train_loader, val_loader, test_loader, self.class_names