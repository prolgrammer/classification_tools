import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
from src.model import ToolClassifier
from src.inference import predict_image
from src.dataset import get_data_loaders
from src.train import train_model
from src.visualize import plot_metrics, plot_confusion_matrix


class ToolClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Construction Tool Classifier")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.classes = None
        self.history = []

        self.setup_gui()

    def setup_gui(self):
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        btn_load = tk.Button(self.root, text="Load Image", command=self.load_image)
        btn_load.pack(pady=5)

        self.result_label = tk.Label(self.root, text="Prediction: None")
        self.result_label.pack(pady=5)

        self.confidence_label = tk.Label(self.root, text="Confidence: None")
        self.confidence_label.pack(pady=5)

        btn_train = tk.Button(self.root, text="Retrain Model", command=self.retrain_model)
        btn_train.pack(pady=5)

        self.history_text = tk.Text(self.root, height=5, width=50)
        self.history_text.pack(pady=10)

        btn_clear_history = tk.Button(self.root, text="Clear History", command=self.clear_history)
        btn_clear_history.pack(pady=5)

    def load_image(self):
        if not self.model or not self.classes:
            self.result_label.config(text="Model not loaded. Please train the model first.")
            return

        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            class_name, confidence = predict_image(file_path, self.model, self.classes, self.device)
            self.result_label.config(text=f"Prediction: {class_name}")
            self.confidence_label.config(text=f"Confidence: {confidence:.4f}")

            self.history.append(f"Image: {file_path}, Prediction: {class_name}, Confidence: {confidence:.4f}")
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, "\n".join(self.history[-5:]))

    def retrain_model(self):
        data_dir = 'data/construction_tools'
        train_loader, val_loader, self.classes = get_data_loaders(data_dir)

        self.model = ToolClassifier(len(self.classes)).to(self.device)
        model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            train_loader, val_loader, len(self.classes), device=self.device
        )
        self.model = model

        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
        plot_confusion_matrix(self.model, val_loader, self.classes, self.device)

        self.result_label.config(text="Model retrained successfully!")

    def clear_history(self):
        self.history = []
        self.history_text.delete(1.0, tk.END)


def run_gui():
    root = tk.Tk()
    app = ToolClassifierApp(root)
    root.mainloop()