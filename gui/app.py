import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os
from model.network import ToolsClassifier
from model.trainer import ModelTrainer
from data.loader import DatasetLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ToolsClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Construction Tools Classifier")
        self.root.geometry("800x600")

        self.data_dir = "./dataset"  # Укажите путь к датасету
        self.model_path = "tools_classifier.pth"
        self.plot_dir = "plots"
        self.log_dir = "logs"

        self.loader = DatasetLoader(self.data_dir)
        self.train_loader, _, self.test_loader, self.class_names = self.loader.load_data()

        self.model = ToolsClassifier(num_classes=len(self.class_names))
        self.trainer = ModelTrainer(self.model, self.train_loader, self.test_loader, self.class_names)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.setup_gui()

    def setup_gui(self):
        self.image_label = tk.Label(self.root, text="No image loaded")
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="Prediction: None", font=("Arial", 14))
        self.result_label.pack(pady=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Show Metrics", command=self.show_metrics).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Show Logs", command=self.show_logs).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear Image", command=self.clear_image).pack(side=tk.LEFT, padx=5)

        self.log_text = tk.Text(self.root, height=10, width=80)
        self.log_text.pack(pady=10)
        self.log_text.insert(tk.END, "Logs will appear here...\n")
        self.log_text.config(state='disabled')

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((200, 200))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

            if os.path.exists(self.model_path):
                self.trainer.load_model(self.model_path)
                img_tensor = self.transform(Image.open(file_path).convert('RGB')).unsqueeze(0)
                prediction = self.trainer.predict(img_tensor)
                self.result_label.config(text=f"Prediction: {prediction}")
            else:
                messagebox.showwarning("Warning", "Model not trained yet. Please train the model first.")

    def train_model(self):
        self.trainer.train(epochs=10)
        messagebox.showinfo("Info", "Training completed! Model saved.")
        self.show_logs()

    def show_metrics(self):
        plot_path = os.path.join(self.plot_dir, 'training_metrics.png')
        if os.path.exists(plot_path):
            img = mpimg.imread(plot_path)
            plt.figure(figsize=(10, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            messagebox.showwarning("Warning", "Metrics plot not found. Train the model first.")

    def show_logs(self):
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        log_files = sorted([f for f in os.listdir(self.log_dir) if f.startswith('log_')], reverse=True)
        if log_files:
            with open(os.path.join(self.log_dir, log_files[0]), 'r') as f:
                self.log_text.insert(tk.END, f.read())
        else:
            self.log_text.insert(tk.END, "No logs found.")
        self.log_text.config(state='disabled')

    def clear_image(self):
        self.image_label.config(image='', text="No image loaded")
        self.result_label.config(text="Prediction: None")