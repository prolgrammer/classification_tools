import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd


class TestDataset(Dataset):
    def __init__(self, image_dir, csv_file, class_names, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.class_names = class_names  # Принимаем class_names извне
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Чтение CSV с проверкой
        try:
            self.annotations = pd.read_csv(csv_file, header=None,
                                           names=['Label', '_1', '_2', '_3', '_4', 'Id', '_5', '_6'])
            print("CSV успешно загружен. Пример данных:")
            print(self.annotations.head(3))
        except Exception as e:
            raise ValueError(f"❌ Ошибка чтения CSV: {e}")

        # Фильтрация данных
        initial_count = len(self.annotations)
        self.annotations = self.annotations[['Id', 'Label']].drop_duplicates('Id')
        print(f"Оставлено {len(self.annotations)}/{initial_count} записей после очистки")

        # Нормализация меток
        self.annotations['Label'] = self.annotations['Label'].str.lower().str.replace(' ', '')
        valid_labels = set(self.class_names)  # Используем переданные class_names

        invalid = set(self.annotations['Label']) - valid_labels
        if invalid:
            print(f"Удалены некорректные метки: {invalid}")
            self.annotations = self.annotations[self.annotations['Label'].isin(valid_labels)]

        # Фильтрация файлов, которые существуют
        valid_files = []
        valid_labels = []
        for idx, row in self.annotations.iterrows():
            img_path = os.path.join(self.image_dir, row['Id'])
            if os.path.exists(img_path):
                valid_files.append(row['Id'])
                valid_labels.append(row['Label'])
            else:
                print(f"Пропущен файл (не существует): {img_path}")

        if not valid_files:
            raise ValueError("❌ Нет доступных файлов после фильтрации. Проверьте пути и CSV.")

        self.image_files = valid_files
        self.labels = valid_labels
        print(f"Оставлено {len(self.image_files)} файлов после проверки существования")

        print("\nРаспределение меток:")
        print(pd.Series(self.labels).value_counts())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            raise

        label_name = self.labels[idx]
        label = self.label_to_idx[label_name]

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

        # Нормализация имен классов
        train_classes = [c.lower().replace('screw driver', 'screwdriver').replace('tool box', 'toolbox') for c
                         in train_dataset.classes]
        val_classes = [c.lower().replace('screw driver', 'screwdriver').replace('tool box', 'toolbox') for c in
                       val_dataset.classes]

        # Определяем валидные классы
        valid_classes = ['gasolinecan', 'hammer', 'pliers', 'rope', 'screwdriver', 'toolbox', 'wrench']
        train_classes = [c for c in train_classes if c in valid_classes]
        val_classes = [c for c in val_classes if c in valid_classes]

        # Проверка согласованности классов
        all_classes = sorted(valid_classes)  # Используем только валидные классы
        print(f"Valid classes: {all_classes}")

        # Загрузка тестового датасета
        test_dataset = TestDataset(test_dir, csv_file, class_names=all_classes, transform=self.test_transform)

        # Приведение классов к единому списку
        train_dataset.classes = all_classes
        val_dataset.classes = all_classes
        test_dataset.class_names = all_classes

        # Обновление маппинга меток
        class_to_idx = {name: idx for idx, name in enumerate(all_classes)}
        train_dataset.class_to_idx = class_to_idx
        val_dataset.class_to_idx = class_to_idx
        test_dataset.label_to_idx = class_to_idx

        # Фильтрация данных в ImageFolder (удаление некорректных меток)

        def filter_dataset(dataset, valid_idx):
            valid_samples = [(path, idx) for path, idx in dataset.samples if idx in valid_idx]
            dataset.samples = valid_samples
            dataset.targets = [idx for _, idx in valid_samples]
            return dataset

        train_dataset = filter_dataset(train_dataset, list(class_to_idx.values()))
        val_dataset = filter_dataset(val_dataset, list(class_to_idx.values()))

        # DataLoader'ы
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print("Unique classes in train:", train_dataset.classes)
        print("Unique classes in val:", val_dataset.classes)
        print("Unique classes in test:", test_dataset.class_names)
        print("Total unique classes:", all_classes)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        return train_loader, val_loader, test_loader, all_classes