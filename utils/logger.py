import os
import logging
from datetime import datetime


class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Создаем имя файла лога с временной меткой
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"log_{timestamp}.txt")

        # Настраиваем логгер
        self.logger = logging.getLogger('ToolsClassifier')
        self.logger.setLevel(logging.INFO)

        # Формат логов
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Файловый обработчик
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(self, message):
        self.logger.info(message)